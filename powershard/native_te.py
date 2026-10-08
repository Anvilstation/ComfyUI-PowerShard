"""Распределённый native text encoder ComfyUI (LTX-2: Gemma 3 12B / Gemma 4 + text projection) в workers.

Идея: тот же класс энкодера и токенайзер, что выбирает ComfyUI (comfy.sd.load_text_encoder_state_dicts),
но без весов на host. Класс берётся «перехватом» clip_target на meta state dict (заголовки safetensors,
реальными читаются только маленькие U8 тензоры токенайзера); в worker модель строится на meta,
соответствие «параметр -> (файл, ключ)» записывается pre-hook-ами load_state_dict при прогоне родного
model.load_sd на meta-тензорах (ключи переименовывает сам ComfyUI), затем веса грузятся локальными
строками FSDP2 Shard(0) (любой формат ComfyUI -> BF16), вычисление FP32 (manual_cast), как cpu_fp32, но на V100.
Последовательность короткая (<=1024 токенов): все rank считают одно и то же (FSDP-only).
"""
import functools
import math
import time
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn

TOKENIZER_KEYS = ("spiece_model", "tokenizer_json", "tokenizer", "gemma_spiece_model", "jina_spiece_model",
                  "yue2_tokenizer_json")
SMALL_REAL_BYTES = 64 * 2**20
SAFETENSORS_DTYPES = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64",
                      "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "U8": "uint8", "I8": "int8",
                      "I16": "int16", "I32": "int32", "I64": "int64", "BOOL": "bool"}
SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
NUMERIC_TAG = "native-te-fp32-compute-v1"


def header_state_dict(path):
    """{ключ: meta tensor} по заголовку; реальные — только маленькие служебные тензоры (токенайзер, маркеры)."""
    from safetensors import safe_open
    from .wan_config import read_safetensors_header, DTYPE_BYTES
    from .quant_formats import annotate_quantized, read_quant_json
    path, _, metadata, header, data_start = read_safetensors_header(path)
    # Таблица как у WanCheckpoint: имя -> desc(key); веса comfy_quant (nvfp4/mxfp8/int8/int4) — логическая форма,
    # чтобы ComfyUI выбрал верный класс энкодера и load_sd не споткнулся о упакованные формы.
    tables = {key: dict(desc, key=key) for key, desc in header.items()}
    annotate_quantized(tables, lambda name: read_quant_json(path, data_start, header[name]))
    sd, real = {}, []
    for key, desc in tables.items():
        count = math.prod(desc.get("stored_shape", desc["shape"]))
        size = count * DTYPE_BYTES[desc["dtype"]]
        if "comfy_quant" in desc:
            sd[key] = torch.empty(desc["shape"], dtype=torch.bfloat16, device="meta")
        elif key in TOKENIZER_KEYS or (desc["dtype"] == "U8" and size <= SMALL_REAL_BYTES) or count <= 64:
            real.append(key)
        else:
            sd[key] = torch.empty(desc["shape"], dtype=getattr(torch, SAFETENSORS_DTYPES[desc["dtype"]]), device="meta")
    if real:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in real:
                sd[key] = f.get_tensor(key)
    return sd, tables, metadata


def storage_report(header):
    counts = {}
    for key, desc in header.items():
        counts[desc["dtype"]] = counts.get(desc["dtype"], 0) + math.prod(desc["shape"])
    unknown = [k for k, d in header.items() if k.endswith(".weight") and d["dtype"] in ("U8", "I8")
               and len(d["shape"]) == 2 and "comfy_quant" not in d and math.prod(d["shape"]) > 2**20]
    if unknown:
        raise ValueError(f"Text encoder: упакованный вес без comfy_quant metadata ({unknown[0]} ...) — формат неизвестен")
    return counts


def capture_clip_target(state_dicts, clip_type):
    """Выбор класса энкодера/токенайзера самим ComfyUI: перехват конструктора comfy.sd.CLIP."""
    import comfy.sd

    class _Captured(Exception):
        pass
    holder = {}
    original = comfy.sd.CLIP

    class _Capture:
        def __init__(self, target=None, embedding_directory=None, parameters=0, tokenizer_data={}, state_dict=[],
                     model_options={}, disable_dynamic=False, **kwargs):
            holder.update(target=target, tokenizer_data=dict(tokenizer_data), model_options=dict(model_options),
                          parameters=parameters)
            raise _Captured()
    comfy.sd.CLIP = _Capture
    try:
        comfy.sd.load_text_encoder_state_dicts(state_dicts, clip_type=clip_type, model_options={})
    except _Captured:
        pass
    finally:
        comfy.sd.CLIP = original
    if "target" not in holder:
        raise RuntimeError("ComfyUI не выбрал text encoder для этих файлов")
    return holder


def prepare_state_dicts(paths):
    """meta state dicts без comfy_quant metadata (все форматы деквантует загрузчик PowerShard)."""
    sds, headers = [], []
    for path in paths:
        sd, header, _ = header_state_dict(path)
        storage_report(header)
        for key in [k for k in sd if k.endswith(".comfy_quant")]:
            sd.pop(key)
        sds.append(sd)
        headers.append(header)
    return sds, headers


def clip_type_value(name):
    import comfy.sd
    return getattr(comfy.sd.CLIPType, name.upper())


# ----------------------------------------------------------------- worker side
def map_parameters(model, state_dicts, identities):
    """Прогон родного model.load_sd на meta-тензорах: какие тензоры файлов попали в какие параметры."""
    seen = {}

    def install():
        handles = []
        for name, module in model.named_modules():
            def hook(module_, state_dict, prefix, local_metadata, strict, missing, unexpected, errors, _name=name):
                for kind in ("_parameters", "_buffers"):
                    for leaf, value in getattr(module_, kind).items():
                        if value is None:
                            continue
                        source = state_dict.get(prefix + leaf)
                        if source is not None and id(source) in identities:
                            seen[(_name + "." if _name else "") + leaf] = identities[id(source)]
            handles.append(module.register_load_state_dict_pre_hook(hook))
        return handles
    # Проход 1 может создать модули (LTXAV compat-коннекторы); проход 2 — с hooks на всех модулях.
    for sd in state_dicts:
        model.load_sd(sd)
    if getattr(model, "compat_mode", False):
        model.enable_compat_mode = lambda: None
    handles = install()
    try:
        for sd in state_dicts:
            model.load_sd(sd)
    finally:
        for handle in handles:
            handle.remove()
    return seen


class TEEntrypoint(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, tokens):
        if command != "encode":
            raise ValueError(f"Неизвестный вызов text encoder: {command}")
        return self.network.encode_token_weights(tokens)


def te_units(model):
    """ModuleList элементы, Embedding и крупные Linear вне них — отдельные FSDP units; цепочка — слои LLM."""
    units, chain, taken = [], [], []
    for name, module in model.named_modules():
        if any(name.startswith(t + ".") or name == t for t in taken):
            continue
        if isinstance(module, nn.ModuleList):
            items = [m for m in module if any(True for _ in m.parameters())]
            if items:
                units += items
                taken.append(name)
                if len(items) > len(chain):
                    chain = items
        elif isinstance(module, nn.Embedding) or (isinstance(module, nn.Linear) and module.weight.numel() >= 2**26):
            units.append(module)
            taken.append(name)
    return units, chain


def load_te_local(root, mapping, paths, headers, device, rank, world, cpu_offload, managed_pool, computed=None):
    """Локальные строки Shard(0) из нескольких файлов; любой формат ComfyUI -> BF16 на GPU rank."""
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    from .config import shard_bounds
    with managed_pool.scope() if managed_pool else nullcontext():
        root.to_empty(device=torch.device("cpu") if cpu_offload else device)
    work = device if device.type == "cuda" else torch.device("cpu")
    evidence, zero_init = [], []
    timing = dict(read_s=0., convert_s=0., store_s=0.)
    tick = time.perf_counter
    handles = [safe_open(str(p), framework="pt", device="cpu") for p in paths]
    with torch.no_grad():
        for name, param in root.named_parameters():
            path = name.removeprefix("network.")
            if not isinstance(param, DTensor) or tuple(p.dim for p in param.placements) != (0,):
                raise RuntimeError(f"Параметр text encoder не FSDP Shard(0): {path}")
            a, b = shard_bounds(param.shape[0], rank, world)
            local = param.to_local()
            if path.endswith(".qbytes"):
                from .quant_linear import build_bytes
                owner = root.get_submodule(name[:-len(".qbytes")])
                table = owner._ps_qplan.source["table"]
                index = next(i for i, h in enumerate(headers) if h is table)
                started = tick()
                full, _ = build_bytes(owner, handles[index], work)
                if local.shape != full[a:b].shape:
                    raise RuntimeError(f"FSDP shape QuantLinear {path}: {local.shape} != {full[a:b].shape}")
                local.copy_(full[a:b])
                timing["convert_s"] += tick() - started
                evidence.append(dict(name=path, rows=[a, b], local_bytes=local.numel(), pinned=local.is_pinned(),
                                     source=f"{Path(paths[index]).name}:{owner._ps_qplan.source['key']} -> "
                                            f"{owner._ps_qplan.fmt}"))
                del full
                continue
            source = mapping.get(path)
            if source is None:
                local.zero_()
                zero_init.append(path)
                continue
            index, key = source
            desc = headers[index][key]
            if tuple(desc["shape"]) != tuple(param.shape):
                raise ValueError(f"Text encoder: форма {key} {desc['shape']} != {list(param.shape)}")
            started = tick()
            if "comfy_quant" in desc:
                # nvfp4 / mxfp8 / int8(ConvRot) / int4 / fp8: родной код ComfyUI, затем свои строки.
                from .quant_formats import dequantize_module, module_entries
                module = key[:-len(".weight")]
                full = dequantize_module(handles[index], module_entries(headers[index], module), module,
                                         desc["comfy_quant"], list(param.shape), work,
                                         torch.float32 if local.dtype == torch.float32 else torch.bfloat16)
                piece = full[a:b]
                del full
            else:
                piece = handles[index].get_slice(key)[a:b]
            timing["read_s"] += tick() - started
            started = tick()
            if work.type == "cuda":
                piece = piece.to(work)
            if "comfy_quant" not in desc and desc["dtype"] in ("F8_E4M3", "F8_E5M2"):
                piece = piece.float()
                module = key[:-len(".weight")] if key.endswith(".weight") else key
                scale_key = next((module + s for s in SCALE_SUFFIXES if module + s in headers[index]), None)
                if scale_key is not None:
                    scale = handles[index].get_tensor(scale_key).float().to(piece.device)
                    if scale.numel() > 1:
                        scale = scale.reshape(-1)[a:b].reshape((b - a,) + (1,) * (piece.ndim - 1))
                    piece = piece * (scale.reshape(-1)[0] if scale.numel() == 1 else scale)
            if piece.is_floating_point():
                check = piece.float()
                if not torch.isfinite(check).all():
                    raise FloatingPointError(f"Не конечные веса text encoder: {key}")
                if local.dtype == torch.float16 and check.numel() and check.abs().max() > 65504:
                    raise FloatingPointError(f"Вес text encoder вне диапазона FP16: {key}")
                del check
            timing["convert_s"] += tick() - started
            started = tick()
            if local.shape != piece.shape:
                raise RuntimeError(f"FSDP shape text encoder: {key}: {local.shape} != {piece.shape}")
            local.copy_(piece.to(dtype=local.dtype))
            timing["store_s"] += tick() - started
            evidence.append(dict(name=path, rows=[a, b], local_bytes=local.numel() * local.element_size(),
                                 pinned=local.is_pinned(), source=f"{Path(paths[index]).name}:{key}"))
        for name, buf in list(root.named_buffers()):
            path = name.removeprefix("network.")
            source = mapping.get(path)
            parent, _, leaf = name.rpartition(".")
            target = torch.empty(buf.shape, dtype=buf.dtype, device=device)
            if source is None:
                if path not in (computed or {}):
                    raise RuntimeError(f"Buffer text encoder без источника: {path}")
                target.copy_(computed[path].to(dtype=buf.dtype))      # вычислен в __init__ (to_empty его стёр)
            else:
                target.copy_(handles[source[0]].get_tensor(source[1]).to(dtype=buf.dtype))
            root.get_submodule(parent)._buffers[leaf] = target
    if managed_pool is not None:
        managed_pool.assert_parameters(root)
    if evidence:
        evidence[0]["load_timing"] = dict(timing, zero_initialized=zero_init[:32], zero_initialized_count=len(zero_init))
    return evidence, zero_init


class NativeTEBackend:
    def __init__(self, paths, clip_type, config, device, managed_pool=None, local_state=None, options=None):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        import comfy.ops
        from .fsdp_backend import assert_sharded, memory, sync
        from .wan_backend import wrap_fsdp_units, load_wan_local
        self.device, self.config, self.managed_pool, self.slot = device, config, managed_pool, "te"
        self.paths = [str(p) for p in paths]
        sds, headers = prepare_state_dicts(self.paths)
        identities = {id(v): (i, k) for i, sd in enumerate(sds) for k, v in sd.items()}
        holder = capture_clip_target(sds, clip_type_value(clip_type))
        target = holder["target"]
        # Как comfy.sd.CLIP: clip(**target.params, device, dtype, model_options). Без torch.device("meta") контекста:
        # параметры создаются на meta (device передан явно), а вычисляемые в __init__ буферы (RoPE inv_freq Gemma 4)
        # остаются настоящими CPU-тензорами — их значения сохраняются ниже.
        params = dict(getattr(target, "params", {}) or {})
        model_options = dict(holder.get("model_options") or {})
        model_options.pop("quantization_metadata", None)
        model_options["custom_operations"] = comfy.ops.manual_cast
        params.update(device="meta", dtype=torch.float16, model_options=model_options)
        model = target.clip(**params)
        mapping = map_parameters(model, sds, identities)
        # BF16 storage (без потерь для bf16-файлов и без риска выхода за диапазон FP16 у Gemma; на V100 тоже —
        # manual_cast приводит веса к FP32 входа), FP32 вычисление.
        dropped = []
        for module_name, module in list(model.named_modules()):
            for leaf, p in list(module._parameters.items()):
                if p is None:
                    continue
                full = (module_name + "." if module_name else "") + leaf
                if full not in mapping and leaf == "logit_scale":
                    module._parameters.pop(leaf)
                    dropped.append(full)
                    continue
                dtype = torch.bfloat16 if p.is_floating_point() else p.dtype
                module._parameters[leaf] = nn.Parameter(torch.empty(p.shape, dtype=dtype, device="meta"),
                                                        requires_grad=False)
        # Embedding не переопределяется: comfy manual_cast ищет строки в FP16 таблице и приводит результат
        # (без FP32 копии таблицы), а ScaledEmbedding Gemma умножает на sqrt(hidden).
        computed = {}
        for name, buf in model.named_buffers():
            if name not in mapping:
                if buf.device.type == "meta":
                    raise RuntimeError(f"Text encoder: buffer {name} без источника в файлах и без вычисленного значения")
                computed[name] = buf.detach().clone()
        root = TEEntrypoint(model)
        root.eval().requires_grad_(False)
        model.execution_device = device
        for child in model.modules():
            if hasattr(child, "execution_device") and child is not model:
                child.execution_device = device
        self.world = dist.get_world_size()
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        # Хранение/вычисление Linear слоёв LLM: как в файле, квантование при загрузке или деквантование (FP32 compute).
        from safetensors import safe_open
        from .quant_linear import install_quant_linears
        options = dict(options or {})
        weight_format, compute = options.get("weight_format", "dequantize"), options.get("compute", "auto")
        _, layer_chain = te_units(model)
        module_names = {id(m): n for n, m in model.named_modules()}
        scope = tuple(module_names[id(m)] + "." for m in layer_chain) or None

        def name_map(param):
            src = mapping.get(param)
            return (headers[src[0]], src[1]) if src else (None, None)

        def open_file(table, key):
            index = next(i for i, h in enumerate(headers) if h is table)
            return safe_open(str(self.paths[index]), framework="pt", device="cpu")
        self.quant_report = install_quant_linears(model, None, weight_format, compute,
                                                  None if weight_format == "as_file" else scope, torch.bfloat16, device,
                                                  open_file=open_file, name_map=name_map, follow_input=True)
        units, chain = te_units(model)
        self.units = wrap_fsdp_units(root, units, chain, mesh, config)
        if local_state is not None:
            self.shards, _ = load_wan_local(root, None, device, dist.get_rank(), self.world, config.cpu_offload, {},
                                            managed_pool, local_state)
            zero_init = []
        else:
            self.shards, zero_init = load_te_local(root, mapping, self.paths, headers, device, dist.get_rank(),
                                                   self.world, config.cpu_offload, managed_pool, computed)
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
        self.patch_fingerprint = NUMERIC_TAG
        self.identity = dict(files=[str(Path(p).name) for p in self.paths], encoder=type(model).__name__,
                             clip_type=clip_type, mapped_parameters=len(mapping), zero_initialized=len(zero_init),
                             dropped=dropped)
        self.zero_initialized = zero_init

    def call(self, command, args, kwargs):
        from .fsdp_backend import assert_sharded, memory, sync
        if command != "encode":
            raise ValueError(f"Text encoder worker: неизвестная команда {command}")
        tokens = args[0]
        if self.zero_initialized and any(isinstance(t, (list, tuple)) and t and isinstance(t[0], dict)
                                         for rows in (tokens.values() if isinstance(tokens, dict) else [tokens])
                                         for row in rows for t in row):
            raise ValueError("Text encoder: изображения в промпте требуют vision-веса, которых нет в файле: "
                             + ", ".join(self.zero_initialized[:4]))
        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode(False), torch.no_grad():
            try:
                result = self.root(command, tokens)
                assert_sharded(self.root, self.config.cpu_offload if self.device.type == "cuda" else None)
            finally:
                for unit in self.units:
                    unit.reshard()
        sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.float().cpu() if x.is_floating_point() else x.cpu()
            if isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(to_cpu(v) for v in x)
            return x
        result = to_cpu(tuple(result))
        if not torch.isfinite(result[0]).all():
            raise FloatingPointError("Text encoder: non-finite conditioning")
        metrics = {"role": "native_te", "expert": "te", "forward_s": time.perf_counter() - start,
                   "memory": memory(self.device), "world_size": self.world, "checkpoint": self.identity,
                   "compute": "FP32 (BF16 storage, manual_cast)",
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
class TEProxy(nn.Module):
    """cond_stage_model без весов: encode_token_weights -> RPC в workers (+ общий conditioning cache)."""

    def __init__(self, session, after_encode, files):
        super().__init__()
        from .wan_text import shared_cache
        self.session, self.after_encode, self.files = session, after_encode, list(files)
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

    def generate(self, *args, **kwargs):
        raise ValueError("PowerShard distributed text encoder: генерация текста (TextGenerate) не поддерживается; "
                         "используйте native loader для prompt enhancement")

    def encode_token_weights(self, tokens):
        import comfy.model_management
        from .conditioning_cache import content_hash
        start = time.perf_counter()
        options = {k: v for k, v in self.clip_options.items() if k != "execution_device"}
        if options.get("layer") not in (None, "last", "all") or options.get("projected_pooled") is not None:
            raise ValueError(f"Distributed text encoder: clip options {options} не поддерживаются")
        stamps = [(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in self.files]
        key = content_hash(dict(tokens=tokens, files=stamps, numeric=NUMERIC_TAG, role="native_te"))
        result = self.cache.get(key)
        hit = result is not None
        if not hit:
            result = self.session.call("encode", (tokens,), {},
                                       cancel=comfy.model_management.throw_exception_if_processing_interrupted)
            self.cache.put(key, result)
            if self.after_encode == "release_now":
                from .wan_text import release_text_encoder
                release_text_encoder(self.session, "release_now")
        self.last_encoding = dict(cache_hit=hit, encoding_wall_s=time.perf_counter() - start, cache=self.cache.report(),
                                  after_encode=self.after_encode)
        return tuple(result)


@functools.lru_cache(maxsize=None)
def _patcher_class():
    import comfy.model_patcher

    class DistributedTEPatcher(comfy.model_patcher.ModelPatcher):
        @property
        def session(self):
            return self.model.session

        def validate(self):
            from .wire import has_effect
            if (self.patches or self.hook_patches or self.weight_wrapper_patches or self.injections or self.forced_hooks
                    or self.additional_models or self.object_patches):
                raise ValueError("Distributed text encoder: LoRA/weight/object/hooks patches не переносятся в workers")
            if any(v for family in (self.wrappers, self.callbacks) for groups in family.values() for v in groups.values()):
                raise ValueError("Distributed text encoder: сторонние callbacks/wrappers не переносятся в workers")
            if has_effect(self.model_options):
                raise ValueError("Distributed text encoder: model_options patches не переносятся в workers")

        def add_patches(self, patches, *args, **kwargs):
            if patches:
                raise ValueError("LoRA для distributed text encoder не реализована")
            return []

        def clone(self, *args, **kwargs):
            other = super().clone(*args, **kwargs)
            proxy = TEProxy(self.session, self.model.after_encode, self.model.files)
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
            return 0

        def model_size(self):
            return 0

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            self.patch_model(device_to=device_to)
            return 0

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            from .wan_text import release_text_encoder
            release_text_encoder(self.session, self.model.after_encode)
            return 0

        def unpatch_model(self, device_to=None, unpatch_weights=True):
            from .wan_text import release_text_encoder
            release_text_encoder(self.session, self.model.after_encode)
            return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)
    return DistributedTEPatcher


@functools.lru_cache(maxsize=None)
def _clip_class():
    import comfy.sd

    class DistributedNativeCLIP(comfy.sd.CLIP):
        def clone(self, disable_dynamic=False):
            other = DistributedNativeCLIP(no_init=True)
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
            self.patcher.session.close()

        def state_dict_for_saving(self):
            raise ValueError("Distributed text encoder не держит весов в host; используйте исходные файлы")

        def get_sd(self):
            return self.state_dict_for_saving()
    return DistributedNativeCLIP


def load_distributed_te(paths, clip_type, config, after_encode="release", report_dir=None, embedding_directory=None,
                        weight_format="dequantize", compute="auto"):
    from dataclasses import replace
    import comfy.sd
    from .devices import resolve_gpu_selection
    from .patch_config import H3PatchConfig
    from .wan_runtime import reusable_wan_session
    if after_encode not in ("release", "release_now", "keep_ram"):
        raise ValueError("after_encode: release / release_now / keep_ram")
    from .quant_formats import validate_quant_choice
    validate_quant_choice(weight_format, compute)
    if after_encode == "keep_ram" and not config.cpu_offload:
        raise ValueError("after_encode=keep_ram требует weight_placement=cpu (shards в pinned RAM)")
    paths = [str(Path(p).resolve()) for p in paths]
    sds, headers = prepare_state_dicts(paths)
    holder = capture_clip_target(sds, clip_type_value(clip_type))
    target = holder["target"]
    tokenizer = target.tokenizer(embedding_directory=embedding_directory, tokenizer_data=holder["tokenizer_data"])
    config = replace(config, precision="fp16", release_after_sampling=True)
    comfy_path = Path(comfy.sd.__file__).resolve().parents[1]
    role_options = dict(kind="native_te", experts={f"te{i}": p for i, p in enumerate(paths)}, files=paths,
                        clip_type=clip_type, options=dict(weight_format=weight_format, compute=compute),
                        families={"te": getattr(target.clip, "__name__", "te")})
    session = reusable_wan_session(paths[0], config, comfy_path, report_dir, patch=H3PatchConfig(enabled=True),
                                   role_options=role_options)
    selected = resolve_gpu_selection(config.gpu_ids)
    total = sum(math.prod(d["shape"]) * 2 for h in headers for k, d in h.items()
                if d["dtype"] != "U8" or "comfy_quant" in d)
    proxy = TEProxy(session, after_encode, paths)
    clip = _clip_class()(no_init=True)
    clip.patcher = _patcher_class()(proxy, torch.device("cuda", int(selected[0]["user_id"])), torch.device("cpu"), size=0)
    clip.patcher.powershard_memory_plan = dict(fp16_bytes_upper_bound=total, shard_bytes_lower_bound=total // max(1, len(selected)),
                                               encoder=getattr(target.clip, "__name__", "te"))
    clip.cond_stage_model = proxy
    clip.tokenizer = tokenizer
    clip.layer_idx = None
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    clip.tokenizer_options = {}
    return clip
