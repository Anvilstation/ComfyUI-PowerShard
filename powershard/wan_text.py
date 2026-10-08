"""umT5-XXL (Wan text encoder) в пуле PowerShard workers.

Host: настоящий comfy.sd.CLIP с native WanT5Tokenizer, без весов энкодера.
Workers: родной comfy WanT5Model (тот же encode_token_weights, маски, zero_out_masked),
веса FSDP2 Shard(0) по выбранным GPU (VRAM или pinned RAM), вычисление FP32 — как
`cpu_fp32`, но на V100. Последовательность ≤512 токенов: считается одинаково на всех rank
(FSDP-only), sequence parallel не нужен.

После encode workers по умолчанию закрываются (`after_encode=release`): GPU свободны для
генератора, а результат держит conditioning cache host (повтор того же промпта — без запуска).
"""
import functools
import time
from pathlib import Path
import torch
import torch.nn as nn

from .wan_config import T5Checkpoint  # noqa: F401  (torch-free reader в wan_config)

# ----------------------------------------------------------------- worker side
def embedding_forward(self, input, out_dtype=None):
    out = torch.nn.functional.embedding(input.to(self.weight.device), self.weight, self.padding_idx, self.max_norm,
                                        self.norm_type, self.scale_grad_by_freq, self.sparse)
    return out.to(out_dtype) if out_dtype is not None else out


class T5Entrypoint(nn.Module):
    """FSDP root: весь encode одним вызовом (shared embedding и блоки — свои units)."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, tokens):
        if command != "encode":
            raise ValueError(f"Неизвестный вызов umT5: {command}")
        return self.network.encode_token_weights(tokens)


class WanT5Backend:
    def __init__(self, path, config, device, managed_pool=None, local_state=None):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        import comfy.ops
        import comfy.text_encoders.wan as wan_te
        from .fsdp_backend import assert_sharded, memory, sync
        from .wan_backend import wrap_fsdp_units, load_wan_local
        self.device, self.config, self.managed_pool, self.slot = device, config, managed_pool, "t5"
        ckpt = T5Checkpoint(path)
        geometry = ckpt.model_config()
        self.identity = dict(ckpt.identity(), storage=ckpt.storage()["kind"], geometry=geometry)
        with torch.device("meta"):
            # FP16 storage, FP32 вычисление: manual_cast приводит веса к dtype входа (SDClipModel даёт float32).
            net = wan_te.WanT5Model(device="meta", dtype=torch.float16,
                                    model_options={"custom_operations": comfy.ops.manual_cast})
            inner = net.umt5xxl
            inner._parameters.pop("logit_scale", None)  # параметр CLIP-L, в umT5 файле его нет и он не используется
            root = T5Entrypoint(net)
        root.eval().requires_grad_(False)
        inner.execution_device = device
        transformer = inner.transformer
        # Явный lookup по FP16 строкам + приведение результата к out_dtype: то же, что comfy Embedding,
        # без зависимости от его cast-политики для таблицы 256k x 4096.
        import types
        transformer.shared.forward = types.MethodType(embedding_forward, transformer.shared)
        if len(transformer.encoder.block) != geometry["num_layers"]:
            raise ValueError(f"umT5: {geometry['num_layers']} блоков в файле, comfy config {len(transformer.encoder.block)}")
        self.world = dist.get_world_size()
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        blocks = list(transformer.encoder.block)
        self.units = wrap_fsdp_units(root, blocks + [transformer.shared], blocks, mesh, config)
        self.shards, _ = load_wan_local(root, ckpt, device, dist.get_rank(), self.world, config.cpu_offload, {},
                                        managed_pool, local_state, None, (), {}, strict=False)
        self.load_timing = next((x["load_timing"] for x in self.shards if "load_timing" in x), None)
        self.root = root
        sync(device)
        assert_sharded(root, config.cpu_offload if device.type == "cuda" else None)
        self.loaded_memory = memory(device)
        self.shard_bytes = sum(x["local_bytes"] for x in self.shards)
        from .phase_cache import ShardBindings
        self.state_bindings = ShardBindings(root)
        self.progress = None
        self.lora_report = []
        self.patch_fingerprint = "umt5-fp32-compute-v1"

    def call(self, command, args, kwargs):
        from .fsdp_backend import assert_sharded, memory, sync
        if command != "encode":
            raise ValueError(f"umT5 worker: неизвестная команда {command}")
        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode(False), torch.no_grad():
            try:
                result = self.root(command, *args)
                assert_sharded(self.root, self.config.cpu_offload if self.device.type == "cuda" else None)
            finally:
                for unit in self.units:
                    unit.reshard()
        sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # не держать reserved память на cuda:0 до старта генератора
        result = tuple(x.float().cpu() if isinstance(x, torch.Tensor) else
                       ({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in x.items()} if isinstance(x, dict) else x)
                       for x in result)
        cond = result[0]
        if not torch.isfinite(cond).all():
            raise FloatingPointError("umT5: non-finite conditioning")
        metrics = {"role": "wan_t5", "expert": "t5", "forward_s": time.perf_counter() - start,
                   "memory": memory(self.device), "world_size": self.world, "checkpoint": self.identity,
                   "compute": "FP32 (FP16 storage, manual_cast)", "tokens": int(cond.shape[1]) if cond.ndim > 1 else None,
                   "weight_memory": dict(persistent_shard_bytes=self.shard_bytes, placement=self.config.weight_placement)}
        return result, metrics

    def idle(self):
        from .fsdp_backend import memory, sync
        for unit in self.units:
            unit.reshard()
        sync(self.device)
        return dict(after=memory(self.device), cpu_shard_bytes=self.shard_bytes if self.config.cpu_offload else 0)

    def end_run(self):
        return None


# ------------------------------------------------------------------- host side
def _host():
    import comfy.model_patcher
    import comfy.sd
    return comfy.model_patcher, comfy.sd


class WanT5Proxy(nn.Module):
    """cond_stage_model без весов: encode_token_weights -> RPC в workers (+ conditioning cache)."""

    def __init__(self, session, after_encode, cache_mib=256):
        super().__init__()
        self.session, self.after_encode = session, after_encode
        self.clip_options = {}
        self.dtypes = {torch.float32}
        self.device = torch.device("cpu")
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.model_loaded_weight_memory = 0
        self.model_offload_buffer_memory = 0
        self.current_weight_patches_uuid = None
        self.cache = shared_cache()
        self.last_encoding = {}

    def reset_clip_options(self):
        self.clip_options = {}

    def set_clip_options(self, options):
        self.clip_options.update(options)

    def memory_estimation_function(self, tokens, device=None):
        return 0

    def encode_token_weights(self, tokens):
        import comfy.model_management
        from .conditioning_cache import content_hash
        start = time.perf_counter()
        options = {k: v for k, v in self.clip_options.items() if k != "execution_device"}
        if options.get("layer") not in (None, "last") or options.get("projected_pooled") is not None:
            raise ValueError(f"umT5 distributed: clip options {options} не поддерживаются (только последний слой)")
        path = Path(self.session.checkpoint)
        stat = path.stat()
        # Результат не зависит от GPU/placement: ключ — только текст (токены) и файл энкодера. Кэш общий для
        # всех экземпляров ноды, поэтому переживает пересоздание loader и смену PowerShardConfig.
        key = content_hash(dict(tokens=tokens, checkpoint=(str(path), stat.st_size, stat.st_mtime_ns),
                                numeric="umt5-fp32-compute-v1", role="wan_t5"))
        result = self.cache.get(key)
        hit = result is not None
        if not hit:
            result = self.session.call("encode", (tokens,), {},
                                       cancel=comfy.model_management.throw_exception_if_processing_interrupted)
            self.cache.put(key, result)
            if self.after_encode == "release_now":
                release_text_encoder(self.session, "release_now")
        self.last_encoding = dict(cache_hit=hit, encoding_wall_s=time.perf_counter() - start, cache=self.cache.report(),
                                  after_encode=self.after_encode)
        return tuple(result)


_SHARED_CACHE = []


def shared_cache():
    from .conditioning_cache import ConditioningCache
    if not _SHARED_CACHE:
        _SHARED_CACHE.append(ConditioningCache(512 * 2**20))
    return _SHARED_CACHE[0]


def release_text_encoder(session, policy):
    """release / release_now: закрыть workers энкодера; keep_ram: оставить shards в pinned RAM.

    При `release` после encode workers живут до старта генератора PowerShard (он деактивирует
    энкодер), ноды Free VRAM или выгрузки ComfyUI — несколько промптов подряд (positive, negative,
    pose) кодируются одним запуском workers, а не тремя.
    """
    if not session.running:
        return
    if policy == "keep_ram" and session.config.cpu_offload:
        session.idle()
    else:
        session.close()


def check_umt5_geometry(geometry):
    """Структура, а не имя файла: подходит любой umT5-XXL (fp16/bf16/fp8_scaled, любое имя)."""
    expected = dict(vocab_size=256384, d_model=4096, num_layers=24)
    if geometry != expected:
        raise ValueError(f"Это не umT5-XXL (Wan): {geometry}, ожидается {expected}. "
                         "Другие T5 (t5xxl Flux/SD3, umt5 base) используют другой tokenizer/config.")
    return geometry


@functools.lru_cache(maxsize=None)
def _patcher_class():
    model_patcher, _ = _host()

    class WanT5Patcher(model_patcher.ModelPatcher):
        @property
        def session(self):
            return self.model.session

        def validate(self):
            from .wire import has_effect
            if (self.patches or self.hook_patches or self.weight_wrapper_patches or self.injections or self.forced_hooks
                    or self.additional_models or self.object_patches):
                raise ValueError("umT5 distributed: LoRA/weight/object/hooks patches не переносятся в workers")
            if any(v for family in (self.wrappers, self.callbacks) for groups in family.values() for v in groups.values()):
                raise ValueError("umT5 distributed: сторонние callbacks/wrappers не переносятся в workers")
            if has_effect(self.model_options):
                raise ValueError("umT5 distributed: model_options patches не переносятся в workers")

        def add_patches(self, patches, *args, **kwargs):
            if patches:
                raise ValueError("LoRA для umT5 в PowerShard не реализована (Wan LoRA к text encoder не применяются)")
            return []

        def clone(self, *args, **kwargs):
            other = super().clone(*args, **kwargs)
            proxy = WanT5Proxy(self.session, self.model.after_encode)
            proxy.cache = self.model.cache
            proxy.clip_options = dict(self.model.clip_options)
            other.model = proxy
            return other

        def load(self, device_to=None, **kwargs):
            self.validate()
            self.model.device = device_to or self.load_device
            self.model.current_patcher = self
            self.model.model_loaded_weight_memory = 0
            self.model.model_lowvram = False
            self.model.current_weight_patches_uuid = self.patches_uuid

        def loaded_size(self):
            return 0  # веса только в workers; host VRAM не занимается

        def model_size(self):
            return 0

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            self.patch_model(device_to=device_to)
            return 0

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            release_text_encoder(self.session, self.model.after_encode)
            return 0

        def unpatch_model(self, device_to=None, unpatch_weights=True):
            release_text_encoder(self.session, self.model.after_encode)
            return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)
    return WanT5Patcher


@functools.lru_cache(maxsize=None)
def _clip_class():
    _, sd = _host()

    class DistributedWanT5CLIP(sd.CLIP):
        def clone(self, disable_dynamic=False):
            other = DistributedWanT5CLIP(no_init=True)
            other.patcher = self.patcher.clone()
            other.cond_stage_model = other.patcher.model
            other.tokenizer = self.tokenizer
            other.layer_idx = self.layer_idx
            other.tokenizer_options = self.tokenizer_options.copy()
            other.use_clip_schedule = self.use_clip_schedule
            other.apply_hooks_to_conds = self.apply_hooks_to_conds
            return other

        def load_model(self, tokens={}):
            self.patcher.validate()
            return super().load_model(tokens)

        def free(self):
            """Закрыть workers энкодера (conditioning cache host сохраняется)."""
            self.patcher.session.close()

        def state_dict_for_saving(self):
            raise ValueError("Distributed umT5 не держит весов в host; используйте исходный файл")

        def get_sd(self):
            return self.state_dict_for_saving()
    return DistributedWanT5CLIP


def load_wan_t5(path, config, after_encode="release", report_dir=None, embedding_directory=None):
    from dataclasses import replace
    import comfy.text_encoders.wan as wan_te
    from safetensors import safe_open
    from .devices import resolve_gpu_selection
    from .patch_config import H3PatchConfig
    from .wan_config import WanOptions, memory_plan
    from .wan_runtime import reusable_wan_session
    import comfy.sd
    ckpt = T5Checkpoint(path)
    check_umt5_geometry(ckpt.model_config())
    if after_encode not in ("release", "release_now", "keep_ram"):
        raise ValueError("after_encode: release / release_now / keep_ram")
    if after_encode == "keep_ram" and not config.cpu_offload:
        raise ValueError("after_encode=keep_ram требует weight_placement=cpu (shards в pinned RAM)")
    # Энкодер не участвует в LoRA/precision генератора: fp16 storage независимо от precision config.
    config = replace(config, precision="fp16", release_after_sampling=True)
    tokenizer_data = {}
    with safe_open(str(ckpt.path), framework="pt", device="cpu") as f:
        if "spiece_model" in f.keys():
            tokenizer_data["spiece_model"] = f.get_tensor("spiece_model")
    comfy_path = Path(comfy.sd.__file__).resolve().parents[1]
    role_options = dict(kind="t5", experts={"t5": str(ckpt.path)}, options=WanOptions().to_dict(),
                        families={"t5": "umt5_xxl"})
    session = reusable_wan_session(str(ckpt.path), config, comfy_path, report_dir, patch=H3PatchConfig(enabled=True),
                                   role_options=role_options)
    selected = resolve_gpu_selection(config.gpu_ids)
    plan = memory_plan(ckpt, len(selected))
    proxy = WanT5Proxy(session, after_encode)
    clip = _clip_class()(no_init=True)
    clip.patcher = _patcher_class()(proxy, torch.device("cuda", int(selected[0]["user_id"])), torch.device("cpu"), size=0)
    clip.patcher.powershard_memory_plan = plan
    clip.cond_stage_model = proxy
    clip.tokenizer = wan_te.WanT5Tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
    clip.layer_idx = None
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    clip.tokenizer_options = {}
    return clip
