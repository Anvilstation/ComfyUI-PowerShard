"""Worker: любая image-модель ComfyUI (SDXL, Flux, Flux.2, Z-Image, Qwen-Image, MOSS-Image, ...) в FSDP2.

Модель строит сам ComfyUI (model_detection -> model_config -> BaseModel) на meta-устройстве, forward — родной
код модели. PowerShard только
* сопоставляет параметры ключам файла (прогон родного load_model_weights на meta-тензорах с pre-hooks),
* загружает локальные строки Shard(0) (любой формат ComfyUI; LoRA — через comfy.lora, см. comfy_lora.py),
* оборачивает ModuleList-блоки в FSDP units и добавляет параллелизм вычислений (image_parallel.py).
"""
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
REMOTE_KEY = "__powershard_remote__"


def _skip_block(args, extra):
    return args


class ImageEntrypoint(nn.Module):
    """FSDP root: forward и вспомогательные методы модели (preprocess_*) идут через root, чтобы FSDP собрал
    параметры корня (embedders, final layer)."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, args, kwargs, call=None):
        if command == "forward":
            return call(*args, **kwargs) if call is not None else self.network(*args, **kwargs)
        if command == "method":
            name = kwargs.pop("_powershard_method")
            return getattr(self.network, name)(*args, **kwargs)
        raise ValueError(f"Image worker: неизвестная команда {command}")


def build_native(path, storage, compute, device="meta"):
    """(BaseModel на meta, model_config, meta state dict, identities, header, prefix)."""
    import comfy.model_detection
    import comfy.utils
    from .native_te import header_state_dict
    sd, header, metadata = header_state_dict(path)
    for key in [k for k in sd if k.endswith(".comfy_quant")]:
        sd.pop(key)
    identities = {id(v): k for k, v in sd.items()}
    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    local = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if local:
        sd = local
    model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
    if model_config is None:
        from .native_te import comfy_version
        raise ValueError(f"{Path(path).name}: ComfyUI ({comfy_version()}) не распознала модель. Причины: (1) модель "
                         "новее вашей ComfyUI — обновите ComfyUI (MOSS-Image — v0.42+, Flux.3 — v0.41+, Fibo 2 — "
                         "v0.43+); (2) файл в diffusers-формате (transformer/*.safetensors) — нужен repackaged файл "
                         "ComfyUI (Comfy-Org)")
    model_config.set_inference_dtype(storage, None if compute == storage else compute)
    if device == "meta":
        import comfy.ops
        model_config.custom_operations = comfy.ops.manual_cast
    # SD2.x: model_type читает значения одного bias (v-prediction) — его нужен настоящий тензор.
    probe = "output_blocks.11.1.transformer_blocks.0.norm1.bias"
    if probe in sd and sd[probe].device.type == "meta":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            real = f.get_tensor(prefix + probe)
        identities[id(real)] = prefix + probe
        sd[probe] = real
    # DynamicVRAM (comfy-aimdo) создаёт Linear без weight и грузит веса лениво, минуя load_state_dict hooks;
    # для meta-копии и сопоставления ключей нужен обычный путь ComfyUI.
    try:
        import comfy.memory_management as memory_management
    except ImportError:  # ComfyUI до DynamicVRAM
        memory_management = None
    saved = getattr(memory_management, "aimdo_enabled", False)
    if memory_management is not None:
        memory_management.aimdo_enabled = False
    try:
        model = model_config.get_model(sd, "", device=torch.device(device))
    finally:
        if memory_management is not None:
            memory_management.aimdo_enabled = saved
    return model, model_config, sd, identities, header, prefix


def promote_file_buffers_to_meta(network, keys, identities):
    """Move buffers that the checkpoint carries onto meta.

    H3 image-family buffers (adaln_t_table, rope.inv_freq) are registered
    without device= and the image path builds the model on CPU, so they are
    live CPU memory. load_model_weights then copies meta->CPU and raises
    "Cannot copy out of meta tensor". Putting the ones the checkpoint actually
    carries on meta lets the hook map them to their file keys.
    """
    for full, buf in list(network.named_buffers()):
        if buf is None or buf.device.type == "meta":
            continue
        parent, _, leaf = full.rpartition(".")
        holder = network.get_submodule(parent) if parent else network
        if full in keys:
            key = full
        elif "diffusion_model." + full in keys:
            key = "diffusion_model." + full
        else:
            continue
        holder._buffers[leaf] = torch.empty_like(buf, device="meta")
        identities[id(holder._buffers[leaf])] = key


def map_parameters(model, sd, identities):
    """Прогон родного BaseModel.load_model_weights на meta: имя параметра diffusion_model -> ключ файла."""
    seen = {}
    network = model.diffusion_model
    promote_file_buffers_to_meta(network, set(sd), identities)
    handles = []
    for name, module in network.named_modules():
        def hook(module_, state_dict, prefix, local_metadata, strict, missing, unexpected, errors, _name=name):
            for kind in ("_parameters", "_buffers"):
                for leaf, value in getattr(module_, kind).items():
                    if value is None:
                        continue
                    source = state_dict.get(prefix + leaf)
                    if source is not None and id(source) in identities:
                        seen[(_name + "." if _name else "") + leaf] = identities[id(source)]
        handles.append(module.register_load_state_dict_pre_hook(hook))
    try:
        model.load_model_weights(dict(sd), "")
    finally:
        for handle in handles:
            handle.remove()
    return seen


def image_units(network):
    """Элементы ModuleList (блоки DiT, input/output blocks UNet) — FSDP units; цепочка prefetch — самый длинный."""
    units, chain, taken = [], [], []
    for name, module in network.named_modules():
        if any(name.startswith(t + ".") or name == t for t in taken):
            continue
        if isinstance(module, nn.ModuleList):
            items = [m for m in module if any(True for _ in m.parameters())]
            if len(items) >= 2:
                units += items
                taken.append(name)
                if len(items) > len(chain):
                    chain = items
    return units, chain


def load_image_local(root, mapping, path, header, device, rank, world, cpu_offload, managed_pool, computed, lora=None):
    """Локальные строки Shard(0); любой формат файла -> dtype параметра; LoRA сливается в FP32 до записи."""
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    from .config import shard_bounds
    from .native_te import SCALE_SUFFIXES
    with managed_pool.scope() if managed_pool else nullcontext():
        root.to_empty(device=torch.device("cpu") if cpu_offload else device)
    work = device if device.type == "cuda" else torch.device("cpu")
    evidence, zero_init = [], []
    timing = dict(read_s=0., convert_s=0., lora_s=0., store_s=0.)
    tick = time.perf_counter
    with torch.no_grad(), safe_open(str(path), framework="pt", device="cpu") as handle:
        for name, param in root.named_parameters():
            pname = name.removeprefix("network.")
            if not isinstance(param, DTensor) or tuple(p.dim for p in param.placements) != (0,):
                raise RuntimeError(f"Параметр image-модели не FSDP Shard(0): {pname}")
            a, b = shard_bounds(param.shape[0], rank, world)
            local = param.to_local()
            if pname.endswith(".qbytes"):
                from .quant_linear import build_bytes
                owner_name = pname[:-len(".qbytes")]
                owner = root.get_submodule(name[:-len(".qbytes")])
                started = tick()
                delta = None
                if lora and lora.has(owner_name + ".weight"):
                    delta = (lambda key, shape, dev, _n=owner_name + ".weight": lora.full_delta(_n, shape, dev))
                full, _ = build_bytes(owner, handle, work, delta)
                if local.shape != full[a:b].shape:
                    raise RuntimeError(f"FSDP shape QuantLinear {pname}: {local.shape} != {full[a:b].shape}")
                local.copy_(full[a:b])
                timing["convert_s"] += tick() - started
                evidence.append(dict(name=pname, rows=[a, b], local_bytes=local.numel(), pinned=local.is_pinned(),
                                     source=f"{owner._ps_qplan.source['key']} -> {owner._ps_qplan.fmt}"))
                del full
                continue
            key = mapping.get(pname)
            if key is None:
                local.zero_()
                zero_init.append(pname)
                continue
            desc = header[key]
            if tuple(desc["shape"]) != tuple(param.shape):
                raise ValueError(f"Image-модель: форма {key} {desc['shape']} != {list(param.shape)}")
            started = tick()
            if "comfy_quant" in desc:
                from .quant_formats import dequantize_module, module_entries
                module = key[:-len(".weight")]
                full = dequantize_module(handle, module_entries(header, module), module, desc["comfy_quant"],
                                         list(param.shape), work, torch.float32)
                piece = full[a:b]
                del full
            else:
                piece = handle.get_slice(key)[a:b]
            timing["read_s"] += tick() - started
            started = tick()
            if work.type == "cuda":
                piece = piece.to(work)
            if "comfy_quant" not in desc and desc["dtype"] in ("F8_E4M3", "F8_E5M2"):
                piece = piece.float()
                module = key[:-len(".weight")] if key.endswith(".weight") else key
                scale_key = next((module + s for s in SCALE_SUFFIXES if module + s in header), None)
                if scale_key is not None:
                    scale = handle.get_tensor(scale_key).float().to(piece.device)
                    if scale.numel() > 1:
                        scale = scale.reshape(-1)[a:b].reshape((b - a,) + (1,) * (piece.ndim - 1))
                    piece = piece * (scale.reshape(-1)[0] if scale.numel() == 1 else scale)
            timing["convert_s"] += tick() - started
            if lora and lora.has(pname) and piece.is_floating_point():
                started = tick()
                delta = lora.delta(pname, a, b, list(param.shape), device=piece.device)
                piece = piece.float() + delta
                timing["lora_s"] += tick() - started
            if piece.is_floating_point():
                check = piece.float()
                if not torch.isfinite(check).all():
                    raise FloatingPointError(f"Не конечные веса image-модели: {key}")
                if local.dtype == torch.float16 and check.numel() and check.abs().max() > 65504:
                    raise FloatingPointError(f"Вес {key} вне диапазона FP16: выберите weight_dtype=bf16/fp32 в узле "
                                             "«PowerShard: формат весов / вычисление»")
                del check
            started = tick()
            if local.shape != piece.shape:
                raise RuntimeError(f"FSDP shape image-модели: {key}: {local.shape} != {piece.shape}")
            local.copy_(piece.to(dtype=local.dtype))
            timing["store_s"] += tick() - started
            evidence.append(dict(name=pname, rows=[a, b], local_bytes=local.numel() * local.element_size(),
                                 pinned=local.is_pinned(), source=key))
        for name, buf in list(root.named_buffers()):
            pname = name.removeprefix("network.")
            key = mapping.get(pname)
            parent, _, leaf = name.rpartition(".")
            target = torch.empty(buf.shape, dtype=buf.dtype, device=device)
            if key is None:
                if pname not in computed:
                    raise RuntimeError(f"Buffer image-модели без источника: {pname}")
                target.copy_(computed[pname].to(dtype=buf.dtype))
            else:
                target.copy_(handle.get_tensor(key).to(dtype=buf.dtype))
            root.get_submodule(parent)._buffers[leaf] = target
    if managed_pool is not None:
        managed_pool.assert_parameters(root)
    if evidence:
        evidence[0]["load_timing"] = dict(timing, zero_initialized=zero_init[:32], zero_initialized_count=len(zero_init))
    return evidence, zero_init


class ImageBackend:
    def __init__(self, path, config, device, role_options, managed_pool=None, local_state=None):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from safetensors import safe_open
        from .fsdp_backend import assert_sharded, memory, sync
        from .wan_backend import wrap_fsdp_units, load_wan_local
        from .image_parallel import ParallelContext, install
        from .cancel import install_block_hooks
        self.device, self.config, self.managed_pool = device, config, managed_pool
        self.path = str(path)
        opts = dict(role_options.get("options") or {})
        self.storage = DTYPES[opts.get("storage", "fp16")]
        self.compute = DTYPES[opts.get("compute", "fp16")]
        self.parallel = opts.get("parallel", "auto")
        weight_format, compute_mode = opts.get("weight_format", "dequantize"), opts.get("compute_mode", "auto")
        model, model_config, sd, identities, header, prefix = build_native(self.path, self.storage, self.compute)
        self.family = type(model).__name__
        network = model.diffusion_model
        mapping = map_parameters(model, sd, identities)
        # Все параметры — в dtype хранения (FSDP группа однородна; manual_cast приводит к dtype вычисления).
        for module in network.modules():
            for leaf, p in list(module._parameters.items()):
                if p is None:
                    continue
                dtype = self.storage if p.is_floating_point() else p.dtype
                module._parameters[leaf] = nn.Parameter(torch.empty(p.shape, dtype=dtype, device="meta"),
                                                        requires_grad=False)
        computed = {}
        for name, buf in network.named_buffers():
            if name not in mapping:
                if buf.device.type == "meta":
                    raise RuntimeError(f"Image-модель: buffer {name} без источника в файле и без вычисленного значения")
                computed[name] = buf.detach().clone()
        total = sum(p.numel() for p in network.parameters())
        unmapped = [n for n, p in network.named_parameters() if n not in mapping]
        unmapped_numel = sum(p.numel() for n, p in network.named_parameters() if n not in mapping)
        norms = [n for n in unmapped if "norm" in n.rsplit(".", 2)[-2:][0].lower()]
        if norms:
            raise ValueError(f"{Path(self.path).name}: веса нормализации {norms[:4]} не найдены в файле — "
                             "формат файла не поддерживается PowerShard (обнулять их нельзя)")
        if unmapped_numel > 0.01 * total:
            raise ValueError(f"{Path(self.path).name}: {len(unmapped)} параметров модели нет в файле (например "
                             f"{unmapped[:4]}) — файл не в формате ComfyUI или неполный")
        # LoRA: разбор ключей родным comfy.lora на BaseModel (meta).
        self.lora = None
        self.lora_report = []
        loras = list((role_options.get("loras") or {}).get("main", ()))
        if loras and local_state is None:
            from .comfy_lora import ComfyLoraDeltas
            self.lora = ComfyLoraDeltas(model, loras)
            self.lora_report = self.lora.report
        elif loras:
            self.lora_report = [dict(cached=True, **l) for l in loras]
        root = ImageEntrypoint(network)
        root.eval().requires_grad_(False)
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.parallel_ctx = ParallelContext(self.world, self.rank)
        self.sequence_families = install(network, self.parallel_ctx)
        units, chain = image_units(network)
        install_block_hooks(chain or units)
        from .quant_linear import install_quant_linears
        names = {id(m): n for n, m in network.named_modules()}
        scope = tuple(names[id(u)] + "." for u in units)

        def name_map(param):
            key = mapping.get(param)
            return (header, key) if key is not None else (None, None)
        # Linear блоков: как в файле / квантование при загрузке (int8/fp8/nvfp4/...) или dequantize (по умолчанию).
        self.quant_report = install_quant_linears(
            network, None, weight_format, compute_mode, None if weight_format == "as_file" else (scope or None),
            self.compute, device, open_file=lambda table, key: safe_open(self.path, framework="pt", device="cpu"),
            name_map=name_map)
        units, chain = image_units(network)
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        self.units = wrap_fsdp_units(root, units, chain, mesh, config)
        if local_state is not None:
            self.shards, _ = load_wan_local(root, None, device, self.rank, self.world, config.cpu_offload, {},
                                            managed_pool, local_state)
            zero_init = []
        else:
            self.shards, zero_init = load_image_local(root, mapping, self.path, header, device, self.rank, self.world,
                                                      config.cpu_offload, managed_pool, computed, self.lora)
        self.load_timing = next((x["load_timing"] for x in self.shards if "load_timing" in x), None)
        self.root = root
        sync(device)
        assert_sharded(root, config.cpu_offload if device.type == "cuda" else None)
        self.loaded_memory = memory(device)
        self.shard_bytes = sum(x["local_bytes"] for x in self.shards)
        from .phase_cache import ShardBindings
        self.state_bindings = ShardBindings(root)
        self.progress = None
        self.remote = {}           # результаты preprocess_* (большие тензоры остаются в worker)
        self.patch_fingerprint = f"image:{self.family}:{opts.get('storage')}:{opts.get('compute')}:{weight_format}"
        self.identity = dict(file=Path(self.path).name, family=self.family, key_prefix=prefix,
                             storage=str(self.storage), compute=str(self.compute), mapped_parameters=len(mapping),
                             zero_initialized=len(zero_init), sequence_parallel=self.sequence_families,
                             parameters=total)
        self.zero_initialized = zero_init
        del model, sd

    # ------------------------------------------------------------------ RPC
    def _resolve(self, value):
        if isinstance(value, dict):
            if set(value) == {REMOTE_KEY}:
                ident = value[REMOTE_KEY]
                if ident not in self.remote:
                    raise RuntimeError("Image worker: результат preprocess устарел (workers перезапущены) — "
                                       "перезапустите задачу")
                return self.remote[ident]
            return {k: self._resolve(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._resolve(v) for v in value)
        return value

    def _cast(self, value):
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != self.compute \
                and value.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return value.to(self.compute)
        if isinstance(value, dict):
            return {k: self._cast(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._cast(v) for v in value)
        return value

    def call(self, command, args, kwargs):
        from .fsdp_backend import assert_sharded, memory, sync
        from .image_parallel import run_parallel
        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        args, kwargs = self._resolve(list(args)), self._resolve(dict(kwargs))
        stats = {}
        with torch.inference_mode(False), torch.no_grad():
            try:
                if command == "forward":
                    x, timestep = args[0], args[1]
                    rest = self._cast(args[2:])
                    kw = self._cast(kwargs)
                    kw["transformer_options"] = dict(kwargs.get("transformer_options") or {})
                    rope = kw.pop("_powershard_rope_options", None)
                    if rope:
                        kw["transformer_options"]["rope_options"] = rope
                    skip = kw.pop("_powershard_skip_blocks", None)
                    if skip:  # Skip Layer Guidance: блок возвращает вход без изменений (родной patches_replace "dit")
                        kw["transformer_options"]["patches_replace"] = {
                            "dit": {(kind, int(index)): _skip_block for kind, index in skip}}

                    def run(x_, t_, *rest_, **kw_):
                        return self.root("forward", (x_, t_) + tuple(rest_), kw_)
                    result, stats = run_parallel(run, self.parallel_ctx, self.parallel, x,
                                                 (self._cast(x), timestep) + tuple(rest), kw)
                elif command == "method":
                    method_kw = dict(self._cast(kwargs))
                    remote_id = method_kw.pop("_powershard_remote_id", None)
                    if remote_id is None:
                        raise RuntimeError("Image worker: нет _powershard_remote_id (host и workers разных версий)")
                    result = self.root("method", tuple(self._cast(args)), method_kw)
                    result = self._keep_large(result, remote_id)
                else:
                    raise ValueError(f"Image worker: неизвестная команда {command}")
                assert_sharded(self.root, self.config.cpu_offload if self.device.type == "cuda" else None)
            finally:
                for unit in self.units:
                    unit.reshard()
        sync(self.device)

        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.float().cpu() if x.is_floating_point() else x.cpu()
            if isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(to_cpu(v) for v in x)
            return x
        if command == "forward":
            if not torch.isfinite(result).all():
                raise FloatingPointError(f"Image-модель ({self.family}) выдала NaN/Inf в {str(self.compute)}: для "
                                         "bf16-моделей на V100 выберите weight_dtype=auto (вычисление FP32)")
            result = result.float().cpu() if self.rank == 0 else result[:0].float().cpu()
        else:
            result = to_cpu(result)
        metrics = {"role": "image", "expert": "main", "forward_s": time.perf_counter() - start,
                   "memory": memory(self.device), "world_size": self.world, "checkpoint": self.identity,
                   "parallel": stats, "compute": str(self.compute),
                   "weight_memory": dict(persistent_shard_bytes=self.shard_bytes, placement=self.config.weight_placement)}
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result, metrics

    def _keep_large(self, value, remote_id, limit=256 << 20, path=()):
        """Большие тензоры preprocess (MOSS ref K/V) остаются в worker, host получает ссылку. id детерминирован
        (id запроса от host + позиция в результате), поэтому совпадает на всех rank'ах."""
        if isinstance(value, torch.Tensor) and value.numel() * value.element_size() > limit:
            ident = ":".join([remote_id] + [str(i) for i in path])
            while len(self.remote) >= 8:
                self.remote.pop(next(iter(self.remote)))
            self.remote[ident] = value
            return {REMOTE_KEY: ident, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, (list, tuple)):
            return type(value)(self._keep_large(v, remote_id, limit, path + (i,)) for i, v in enumerate(value))
        if isinstance(value, dict):
            return {k: self._keep_large(v, remote_id, limit, path + (k,)) for k, v in value.items()}
        return value

    def idle(self):
        from .fsdp_backend import memory, sync
        for unit in self.units:
            unit.reshard()
        sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return dict(after=memory(self.device), cpu_shard_bytes=self.shard_bytes if self.config.cpu_offload else 0)

    def end_run(self):
        return None


def worker_memory_estimate(parameters, storage_bytes, world):
    return dict(total_bytes=parameters * storage_bytes, shard_bytes=math.ceil(parameters * storage_bytes / world))
