"""Park LOCAL shards in RAM without changing an active FSDP offload policy.

The cached tensors never contain all-gathered parameters. GPU/ATS models are
discarded at a phase boundary and rebuilt from these validated local shards,
not from checkpoint data. CPUOffloadPolicy models already have CPU shards.
"""
import gc
import time
import torch


class ShardBindings:
    """Canonical names before Spectrum gates add a `.block` namespace."""
    def __init__(self, root):
        def binding(name):
            parent, _, leaf = name.rpartition(".")
            return root.get_submodule(parent), leaf
        self.parameters = {name: binding(name) for name, _ in root.named_parameters()}
        self.buffers = {name: binding(name) for name, _ in root.named_buffers()}
        self.linears = {name: module for name, module in root.named_modules()
                        if hasattr(module, "_ps_weight_bounds")}
        self.attention = {name: module for name, module in root.named_modules()
                          if hasattr(module, "_ps_qk_scales")}

    def capture(self):
        def cpu_snapshot(bindings, local):
            saved = {}
            for name, (module, leaf) in bindings.items():
                tensor = getattr(module, leaf)
                if local:
                    if not hasattr(tensor, "to_local"):
                        raise RuntimeError("Parking requires a resharded DTensor: " + name)
                    tensor = tensor.to_local()
                # Owned storage: no GPU reference, mmap, or FSDP object in RAM cache.
                saved[name] = tensor.detach().to("cpu", copy=True)
            return saved
        return LocalShardState(cpu_snapshot(self.parameters, True),
                               cpu_snapshot(self.buffers, False),
                               {n: tuple(m._ps_weight_bounds) for n, m in self.linears.items()},
                               {n: tuple(m._ps_qk_scales) for n, m in self.attention.items()})


class LocalShardState:
    def __init__(self, parameters, buffers, weight_bounds, qk_scales=None):
        self.parameters, self.buffers, self.weight_bounds = parameters, buffers, weight_bounds
        self.qk_scales = qk_scales or {}
        if any(t.device.type != "cpu" for t in (*parameters.values(), *buffers.values())):
            raise ValueError("A parked shard cache must be CPU-only")

    @property
    def parameter_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.parameters.values())

    @property
    def buffer_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.buffers.values())


class ResidentBackend:
    """Single worker owner. Parking happens between runs, never between steps."""
    def __init__(self, backend, factory, device):
        self._backend, self._factory, self.device = backend, factory, device
        self._cached = None
        self.progress = None

    def __getattr__(self, name):
        # In particular, hasattr(owner, 'spectrum') must not reload a parked model.
        backend = self.__dict__.get("_backend")
        if backend is None:
            raise AttributeError(name)
        return getattr(backend, name)

    def end_run(self):
        if self._backend is None:
            return None
        spectrum = getattr(self._backend, "spectrum", None)
        report = spectrum.report() if spectrum else None
        if spectrum:
            spectrum.clear()
        return report

    def idle(self):
        from .fsdp_backend import assert_sharded, memory, sync
        if self._backend is None:
            return dict(after=memory(self.device), cpu_shard_bytes=self._cached.parameter_bytes,
                        retained="CPU-only local shard cache", already_idle=True)
        if self._backend.config.cpu_offload:
            return self._backend.idle()
        before = memory(self.device)
        start = time.perf_counter()
        self.end_run()
        for unit in self._backend.units:
            unit.reshard()
        sync(self.device)
        assert_sharded(self._backend.root, False if self.device.type == "cuda" else None)
        cached = self._backend.state_bindings.capture()
        if cached.parameter_bytes != self._backend.shard_bytes:
            raise RuntimeError("Parking changed the local shard byte count")
        # Drop the entire FSDP graph; do NOT edit private FSDP policy/storage.
        # GC here is needed for hook/module cycles, but never runs in a block.
        self._backend = None
        gc.collect()
        sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._cached = cached
        return dict(before=before, after=memory(self.device), parking_s=time.perf_counter()-start,
                    cpu_shard_bytes=cached.parameter_bytes, cpu_buffer_bytes=cached.buffer_bytes,
                    checkpoint_weight_reads=0, retained="CPU-only local shard cache; CUDA/NCCL context may remain")

    def call(self, command, args, kwargs):
        resume_s = 0.
        resumed = self._backend is None
        if resumed:
            start = time.perf_counter()
            backend = self._factory(self._cached)
            # Keep the RAM cache until a complete reconstruction succeeds.
            self._backend = backend
            self._cached = None
            resume_s = time.perf_counter()-start
        self._backend.progress = self.progress
        result, metrics = self._backend.call(command, args, kwargs)
        metrics["phase_cache"] = dict(resumed_from_ram=resumed, resume_from_ram_s=resume_s,
                                      checkpoint_weight_reads_on_resume=0)
        return result, metrics
