"""MiniMax H3 с выбором хранения/вычисления весов (любой формат ComfyUI, native int8/fp8/nvfp4/mxfp8/int4).

Исходный H3 путь (comfy_adapter.load_model / worker.py / H3Backend) не меняется. Здесь — отдельный
loader: та же native MiniMaxH3 модель, attention, FP16 Safe patch, Spectrum и sequence parallel
(H3Backend наследуется целиком: call/plan_memory/idle), но веса читаются общим загрузчиком
PowerShard (любые comfy_quant форматы) и Linear блоков могут храниться квантованными (QuantLinear).
"""
from pathlib import Path
import torch
from .wan_config import WanCheckpoint


class H3Checkpoint(WanCheckpoint):
    """Заголовок H3 c логическими формами квантованных весов (nvfp4/int4 упакованы)."""

    def detect_prefix(self, header):
        for prefix in ("", "model.diffusion_model.", "diffusion_model."):
            if prefix + "video_patch_proj.weight" in header and prefix + "adaln_t_table" in header:
                return prefix
        raise ValueError("Это не native MiniMax H3 checkpoint (нет video_patch_proj.weight / adaln_t_table)")

    def pdd_steps(self):
        """PDD checkpoint: pdd_steps стопок выходных голов (metadata config.transformer.pdd_steps, как в ComfyUI)."""
        import json
        try:
            return int(json.loads(self.metadata.get("config", "{}")).get("transformer", {}).get("pdd_steps", 1))
        except (ValueError, TypeError, AttributeError):
            return 1

    def model_config(self):
        from .checkpoint import infer_h3_config
        conf = infer_h3_config(self.tensors)
        pdd = self.pdd_steps()
        if pdd > 1:
            conf["latents_dim"] //= pdd
            conf["audio_latents_dim"] //= pdd
            conf["pdd_steps"] = pdd
        return conf


def _backend_class():
    from .fsdp_backend import H3Backend

    class QuantH3Backend(H3Backend):
        def __init__(self, checkpoint, config, device, patch=None, attention_policy=None, role_options=None,
                     managed_pool=None, local_state=None):
            import math
            import torch.distributed as dist
            from torch.distributed.device_mesh import init_device_mesh
            from safetensors import safe_open
            from comfy.ldm.minimax.model import MiniMaxH3Model
            from .attention import install_attention, install_sequence, configure_safe_qk
            from .attention_policy import AttentionDispatcher
            from .fp16_safe import apply_fp16_safe
            from .fsdp_backend import Entrypoint, wrap_fsdp, sync, memory, assert_sharded, fsdp_groups
            from .operations import Operations, install_int8
            from .patch_config import H3PatchConfig
            from .quant_linear import install_quant_linears
            from .wan_backend import load_wan_local
            options = dict((role_options or {}).get("options", {}))
            weight_format, compute = options.get("weight_format", "dequantize"), options.get("compute", "auto")
            self.device, self.config, self.managed_pool = device, config, managed_pool
            self.patch = patch or H3PatchConfig()
            ckpt = H3Checkpoint(checkpoint)
            shape = ckpt.model_config()
            if shape.get("pdd_steps", 1) > 1:
                import inspect
                if "pdd_steps" not in inspect.signature(MiniMaxH3Model.__init__).parameters:
                    raise RuntimeError("PDD checkpoint MiniMax H3 требует ComfyUI с поддержкой pdd_steps (v0.42+)")
            # Native int8 H3 (Int8Linear) — только при dequantize-хранении и int8 checkpoint; иначе QuantLinear/деквант.
            quant = ckpt.quantization() if config.precision == "int8_fp16" and weight_format == "dequantize" else {}
            loras = list(((role_options or {}).get("loras") or {}).get("main", ()))
            if loras and quant:
                raise ValueError("LoRA нельзя слить в native INT8 H3 (precision=int8_fp16): выберите precision=fp16 и "
                                 "формат int8 в узле «PowerShard: формат весов / вычисление»")
            self.lora, self.lora_report = None, []
            if loras and local_state is None:
                from .comfy_lora import ComfyLoraDeltas
                self.lora = ComfyLoraDeltas(h3_base_model(shape), loras)
                self.lora_report = self.lora.report
            elif loras:
                self.lora_report = [dict(cached=True, **l) for l in loras]
            self.identity = dict(ckpt.identity(), storage=ckpt.storage()["kind"], weight_format=weight_format,
                                 compute=compute)
            with torch.device("meta"):
                net = MiniMaxH3Model(**shape, dtype=torch.float16, device="meta", operations=Operations)
                quant_map = install_int8(net, quant, config.dequant_rows) if quant else {}
                root = Entrypoint(net)
            root.eval().requires_grad_(False)
            self.attention = AttentionDispatcher(config, attention_policy)
            install_attention(net, config, self.attention)
            from .veda.runtime import VedaRuntime
            self.veda = VedaRuntime(net, device)  # Veda Sparse Attention: включается per-RPC нодой host
            from .cancel import install_block_hooks
            install_block_hooks(list(net.blocks))  # Cancel прерывает шаг между блоками
            cached_qk = ({name.removeprefix("network."): scales for name, scales in local_state.qk_scales.items()}
                         if local_state else None)
            configure_safe_qk(net, ckpt, cached_qk)
            self.tracker = apply_fp16_safe(net, self.patch)
            install_sequence(net, config)
            # H3 attention/dispatcher сертифицированы на FP16: QuantLinear считает в FP16 (native ядра — тоже).
            self.quant_report = install_quant_linears(
                net, ckpt.tensors, weight_format, compute,
                None if weight_format == "as_file" else ("blocks.", "token_refiner.blocks."), torch.float16, device,
                open_file=lambda table, key: safe_open(str(ckpt.path), framework="pt", device="cpu"))
            self.world = dist.get_world_size()
            mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
            self.units = wrap_fsdp(root, mesh, config)
            self.shards, _ = load_wan_local(root, ckpt, device, dist.get_rank(), self.world, config.cpu_offload,
                                            quant_map, managed_pool, local_state, self.lora, (), {}, strict=True)
            self.root = root
            sync(device)
            assert_sharded(root, config.cpu_offload if device.type == "cuda" else None)
            self.loaded_memory = memory(device)
            self.shard_bytes = sum(x["local_bytes"] for x in self.shards)
            if config.cpu_offload and config.pin_memory and any(not x["pinned"] for x in self.shards if x["local_bytes"]):
                raise RuntimeError("CPUOffloadPolicy(pin_memory=True) не создал pinned локальные shards")
            self.buffer_bytes = sum(b.numel() * b.element_size() for b in root.buffers())
            self.group_bytes = [sum(math.prod(p._orig_size) * p.sharded_param.element_size()
                                    for group in fsdp_groups(unit) for p in group.fsdp_params) for unit in self.units]
            self.root_group_bytes = self.group_bytes[-1]
            from .phase_cache import ShardBindings
            self.state_bindings = ShardBindings(root)
            from .telemetry import ForwardLedger
            self.ledger = ForwardLedger()
            names = {id(m): n for n, m in root.named_modules()}
            for unit in self.units:
                self.ledger.attach(names.get(id(unit), "root"), unit, fsdp_groups(unit))
            self.memory_context = {}
            for module in root.modules():
                if hasattr(module, "_ps_mlp_chunk"):
                    module._ps_memory_context = self.memory_context
            from .spectrum import SpectrumEngine, install_spectrum
            from .spectrum_config import SpectrumConfig
            self.spectrum = SpectrumEngine(SpectrumConfig(**(role_options or {}).get("spectrum", {})), device)
            install_spectrum(net, self.spectrum)
            self.load_timing = next((x["load_timing"] for x in self.shards if "load_timing" in x), None)
            self.patch_fingerprint = self.patch.fingerprint() + (":lora" if loras else "")

        def call(self, command, args, kwargs):
            kwargs = dict(kwargs)
            spec = kwargs.pop("_powershard_veda", None)
            if command == "forward":
                self.veda.begin(spec)  # на всех rank одновременно: выбор бэкенда может делать all-reduce
            try:
                result, metrics = super().call(command, args, kwargs)
            finally:
                self.veda.context = {}  # не держать sigmas/layout прошлого шага между RPC
            if command == "forward" and spec:
                metrics["veda"] = self.veda.report()
            return result, metrics

    return QuantH3Backend


def quant_h3_backend():
    return _backend_class()


def h3_base_model(shape):
    """BaseModel MiniMaxH3 на meta — только для разбора ключей LoRA родным comfy.lora (в т.ч. банки PDD-голов)."""
    import comfy.ops
    import comfy.supported_models
    model_config = comfy.supported_models.MiniMaxH3(dict(shape, image_model="minimax_h3"))
    model_config.set_inference_dtype(torch.float16, None)
    model_config.custom_operations = comfy.ops.manual_cast
    return model_config.get_model({}, "", device=torch.device("meta"))


VEDA_KEY = "powershard_veda"


def _host_classes():
    """Proxy/patcher H3 с передачей настроек Veda в RPC (исходные comfy_adapter классы не меняются)."""
    from .comfy_adapter import DiffusionProxy, PowerShardPatcher

    class H3QuantProxy(DiffusionProxy):
        def forward(self, x, timestep, context, control=None, transformer_options=None, **kwargs):
            # DIFFUSION_MODEL wrappers (EasyCache) — на host вокруг RPC, как в родном forward.
            from .accel import run_diffusion_wrappers
            options = dict(transformer_options or {})
            return run_diffusion_wrappers(self._remote_h3, self, (x, timestep, context),
                                          dict(control=control, transformer_options=options, **kwargs), options)

        def _remote_h3(self, x, timestep, context, control=None, transformer_options=None, **kwargs):
            from .accel import HOST_ONLY_TRANSFORMER_KEYS, strip_host_safe_wrappers
            options = dict(transformer_options or {})
            spec = options.pop(VEDA_KEY, None)
            if spec is not None:
                kwargs["_powershard_veda"] = spec
            for key in HOST_ONLY_TRANSFORMER_KEYS:  # Context Windows / EasyCache работают на host
                options.pop(key, None)
            if "wrappers" in options:
                options["wrappers"] = strip_host_safe_wrappers(options["wrappers"])
            return super().forward(x, timestep, context, control=control, transformer_options=options, **kwargs)

    class H3QuantPatcher(PowerShardPatcher):
        def clone(self, *args, **kwargs):
            result = super().clone(*args, **kwargs)
            if hasattr(self, "powershard_h3_shape"):
                result.powershard_h3_shape = dict(self.powershard_h3_shape)
            return result

        def validate(self):
            # Как PowerShardPatcher.validate, но допускает host-only ускорители ComfyUI (Context Windows, EasyCache,
            # LazyCache, CFG-функции) и настройки Veda.
            from .accel import (HOST_ONLY_TRANSFORMER_KEYS, check_easycache_holder, host_safe_model_options,
                                strip_host_safe_wrappers)
            from .host_guard import strip_internal_wrappers
            from .wire import has_effect, validate_options
            if self.patches or self.hook_patches or self.weight_wrapper_patches or self.injections:
                raise ValueError("Родной LoRA loader / hooks не применимы к FSDP H3: подключите «PowerShard: LoRA» "
                                 "ко входу lora «PowerShard MiniMax H3: Loader»")
            if self.forced_hooks is not None or self.additional_models:
                raise ValueError("Дополнительные модели/hooks не поддерживаются")
            if not self.session.config.allow_host_wrappers:
                for family in (strip_host_safe_wrappers(strip_internal_wrappers(self.wrappers)), self.callbacks):
                    if any(v for groups in family.values() for v in groups.values()):
                        raise ValueError("Сторонние callbacks/wrappers: установите allow_host_wrappers=True в "
                                         "PowerShard-конфиге (без этого допущены EasyCache, LazyCache, Context Windows)")
            if set(self.object_patches) - {"model_sampling"}:
                raise ValueError("Разрешён только native model_sampling patch")
            if any(has_effect(v) for k, v in host_safe_model_options(self.model_options).items()
                   if k not in {"transformer_options", "to_load_options"}):
                raise ValueError("Сторонние model_options не поддерживаются")
            to_load = dict(self.model_options.get("to_load_options", {}))
            if "wrappers" in to_load:
                to_load["wrappers"] = strip_internal_wrappers(to_load["wrappers"])
            if has_effect(to_load):
                raise ValueError("Сторонние to_load_options не поддерживаются")
            options = {k: v for k, v in self.model_options.get("transformer_options", {}).items()
                       if k not in HOST_ONLY_TRANSFORMER_KEYS + (VEDA_KEY,)}
            if self.model_options.get("transformer_options", {}).get("easycache") is not None:
                check_easycache_holder(self.model_options["transformer_options"]["easycache"])
            if "wrappers" in options:
                options["wrappers"] = strip_host_safe_wrappers(options["wrappers"])
                if self.session.config.allow_host_wrappers:
                    options.pop("wrappers")
            validate_options(options)

        def with_veda(self, spec):
            result = self.clone()
            options = result.model_options.setdefault("transformer_options", {})
            if spec is None:
                options.pop(VEDA_KEY, None)
            else:
                options[VEDA_KEY] = dict(spec)
            return result
    return H3QuantProxy, H3QuantPatcher


_HOST = []


def host_classes():
    if not _HOST:
        _HOST.extend(_host_classes())
    return tuple(_HOST)


def load_h3(path, config, weight_format="dequantize", compute="auto", report_dir=None, fp16_accumulation=False,
            loras=()):
    """Как comfy_adapter.load_model, но workers — wan_worker (роль h3, kind h3q) с выбором хранения/вычисления."""
    import comfy.model_base
    import comfy.supported_models
    from .comfy_adapter import RemoteH3
    DiffusionProxy, PowerShardPatcher = host_classes()
    from .devices import resolve_gpu_selection
    from .patch_config import H3PatchConfig
    from .source_guard import verify_comfy
    from .wan_config import memory_plan
    from .wan_runtime import reusable_wan_session
    from .quant_formats import validate_quant_choice
    validate_quant_choice(weight_format, compute)
    comfy_path = Path(comfy.model_base.__file__).resolve().parents[1]
    verify_comfy(comfy_path)
    ckpt = H3Checkpoint(path)
    shape = ckpt.model_config()
    if config.precision == "int8_fp16" and weight_format == "dequantize":
        ckpt.quantization()  # native int8 путь: только int8_tensorwise
    model_config = comfy.supported_models.MiniMaxH3(dict(shape, image_model="minimax_h3", disable_unet_model_creation=True,
                                                         dtype=torch.float16))
    model_config.manual_cast_dtype = torch.float16
    model = RemoteH3(model_config, device=torch.device("cpu"))
    loras = [dict(path=str(l.path), strength=float(l.strength)) for l in loras or ()]
    if loras and config.precision == "int8_fp16" and weight_format == "dequantize":
        raise ValueError("LoRA нельзя слить в native INT8 H3 (precision=int8_fp16): выберите precision=fp16 и формат "
                         "int8 в узле «PowerShard: формат весов / вычисление»")
    role_options = dict(kind="h3q", experts={"main": str(ckpt.path)},
                        options=dict(weight_format=weight_format, compute=compute,
                                     fp16_accumulation=bool(fp16_accumulation)))
    if loras:
        role_options["loras"] = {"main": loras}
    session = reusable_wan_session(str(ckpt.path), config, comfy_path, report_dir,
                                   patch=H3PatchConfig(enabled=True, mlp_chunk_mode="off", mlp_chunk_tokens=4096),
                                   role_options=role_options)
    model.diffusion_model = DiffusionProxy(session, shape)
    model.diffusion_model.dtype = torch.float32
    model.eval().requires_grad_(False)
    selected = resolve_gpu_selection(config.gpu_ids)
    plan = memory_plan(ckpt, len(selected))
    size = (plan["shard_bytes_lower_bound"] if config.weight_placement == "gpu" else 0)
    size += (1 + config.prefetch_blocks) * plan["largest_group_bytes_upper_bound"]
    plan["host_gpu_parameter_budget_bytes"] = size
    plan.update(weight_format=weight_format, compute=compute)
    patcher = PowerShardPatcher(model, torch.device("cuda", int(selected[0]["user_id"])), torch.device("cpu"), size=size)
    patcher.powershard_memory_plan = plan
    patcher.powershard_h3_shape = dict(shape)  # геометрия для проверки предиктора Veda на host
    return patcher
