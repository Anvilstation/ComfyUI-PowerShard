# ComfyUI-PowerShard 0.5.3

Distributed MiniMax H3 inference for ComfyUI on a selected set of CUDA GPUs.

[Русская документация](README_RU.md) · [0.5.3 changes](PERFORMANCE_FIX_0_5_3_RU.md) · [Compatibility](COMPATIBILITY.md)

FSDP2 shards weights across the selected devices. Token and Ulysses sequence modes also distribute H3 computation. Native ComfyUI conditioning, samplers, and video/audio VAE nodes remain part of the workflow.

## Current features

- A compact five-field configuration: GPU list, GPU/CPU/ATS weight placement, precision, attention backend, and sequence mode.
- H3 FP16 Safe and INT8 ConvRot storage; separate optional H3/Qwen MLP chunking nodes, disabled by default.
- Ulysses supports non-divisible head counts through padding, including five- and six-GPU configurations.
- Distributed Qwen3-VL conditioning with compatible ComfyUI load/unload and clone behavior.
- `keep_in_memory=true` retains local weight shards in system RAM between tasks for GPU, CPU, and ATS placements, then restores the chosen active placement without rereading checkpoint weights.
- Scoped FlashAttention API compatibility over an installed vLLM provider; no global `sys.path` changes.
- FP32 sequence exchange by default, optional normalized FP16 exchange, explicit prefetch policy, and separate worker/host timing reports.
- Spectrum is an optional approximation that forecasts selected H3 calls. Compare output quality separately.

## Installation and workflows

Use the Python environment of your existing ComfyUI installation. Back up the previous custom node and place this directory at `ComfyUI/custom_nodes/ComfyUI-PowerShard`. Keep one copy of the node, restart ComfyUI, and refresh the browser. Model weights are not included. Do not replace your existing torch, CUDA, NCCL, or custom attention wheel as part of this update.

```bash
python scripts/diagnose.py
python scripts/dependency_plan.py
```

Start with `workflows/fl2va_fp16.ui.json`, or one of the six-device examples `workflows/ac922_6gpu_{gpu,cpu,ats}_{token,ulysses}.ui.json`. Matching API JSON files are included. Select your existing checkpoint and VAE files and the intended GPU list. `all` means all GPUs visible to the ComfyUI process; a nonempty subset is also supported.

Before VAE decoding, `PowerShardRelease` can release active H3/Qwen GPU state while preserving RAM shards. It does not manage the host ComfyUI VAE. ATS applies to distributed worker weights through a scoped managed allocator; it does not automatically offload every host model or tensor. Unsupported ATS capabilities produce an explicit error.

Older workflows require migration when positional widgets have changed. Use `scripts/migrate_workflow.py` and the migration instructions in [README_RU.md](README_RU.md). Explicit old MLP `auto`/`manual` settings are preserved; switch them to `off` if desired.

## Verification

The included 0.5.3 report records **298 CPU/native tests passed and 16 distributed checks skipped** because Gloo transport was unavailable. This validates CPU contracts and small native models, not CUDA/NCCL/POWER9/ATS execution or production output quality. The user's later AC922 runs are separate server evidence; no universal speedup or OOM-free operation is promised.

See [verification metadata](reports/performance-0.5.3/verification.json), [known limitations](KNOWN_LIMITATIONS.md), and [0.5.2 lifecycle fixes](FIXES_0_5_2_RU.md). Remote LoRA/ControlNet support is not implemented.

Historical documentation and test results remain for reference, including [the earlier English README](docs/archive/README_EN_0_5_0rc1.md). They do not describe the current configuration API.

License: [GPL-3.0](LICENSE). Third-party notices are retained in [licenses](licenses).
