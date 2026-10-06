# ComfyUI-PowerShard

**Distributed MiniMax H3 inference in a regular ComfyUI workflow.** PowerShard runs one generation across selected CUDA GPUs. FSDP2 shards the generator weights while native H3 conditioning, samplers, and video/audio VAE nodes remain part of the workflow. A mixed precision path addresses V100 FP16 overflow; INT8 ConvRot storage is retained instead of unpacking the entire model into a persistent FP16 copy.

[Русский](README_RU.md) · [Compatibility](COMPATIBILITY.md) · [Results and test status](BENCHMARKS.md)

## Features

| Area | Implementation |
|---|---|
| GPU selection | Any nonempty subset of visible CUDA devices: `0`, `1,3,5`, `5,2,0`, or `all`. Selection order is preserved. |
| H3 weight sharding | One worker per GPU; FSDP2 FULL_SHARD gathers active block parameters and reshards them after forward. A single GPU has no inter-GPU sharding. |
| MiniMax H3 | FL2VA and Ref2VA Pruned, BF16 source checkpoints with an FP16 runtime path, and INT8 ConvRot. Worker-side FP16 Safe keeps sensitive operations in FP32. |
| Attention | `auto`, PyTorch SDPA, FlashAttention, a custom `vllm_flash_attn` build, SageAttention, and reference/math. An unavailable provider reports why and can fall back with `allow_fallback=true`. |
| Compute distribution | Separate `fsdp2_sequence` mode partitions token/query work. Plain `fsdp2` primarily distributes weight memory. |
| Qwen3-VL-32B CLIP | H3 text/vision conditioning node with FP16/INT8 options, distributed placement, CPU offload, and a bounded RAM cache. |
| Memory policy | CPU offload of local FSDP shards, configurable prefetch, and a `ram_min` profile that favors keeping weights in system RAM. It may reduce speed. |

PowerShard does **not** replace an existing ComfyUI, CUDA, PyTorch, NCCL, or custom kernel wheel. Optional attention packages are loaded only when needed. Model weights are not included.

## Quick start

Use the Python environment of your existing ComfyUI installation. Back up any installed custom node before updating, then place this directory at `ComfyUI/custom_nodes/ComfyUI-PowerShard`. Inspect the environment and dependency plan before installing anything:

```bash
python scripts/diagnose.py --comfy /ABS/ComfyUI --output reports/local-environment.json
python scripts/check_source_requirements.py
python scripts/dependency_plan.py --output-dir reports/local-dependency-plan
```

Install only missing runtime dependencies with that same Python environment if needed. Do not automatically upgrade torch or your custom CUDA wheel. Platform instructions: [ppc64le](docs/INSTALL_PPC64LE.md) and [x86_64](docs/INSTALL_X86_64.md).

From the PowerShard directory, start ComfyUI:

```bash
bash scripts/launch_comfy.sh /ABS/venv/bin/python /ABS/ComfyUI
```

Open **`workflows/fl2va_ram_min_int8.ui.json`** for the minimum VRAM profile or **`workflows/fl2va_fp16.ui.json`** for the basic FP16 path. Select your local H3/Qwen/VAE checkpoints and devices in Distributed Config. The example `0,1,2` is not a GPU count restriction. MODEL flows through **PowerShard MiniMax H3 FP16 Patcher** into the native guider/sampler.

Run a CUDA/NCCL probe on the chosen devices before loading large weights:

```bash
python scripts/probe_devices.py --gpus 0,1,2 --attention-backend sdpa --reports reports/local-probe
```

[Checkpoint metadata](models/CHECKPOINTS_RU.md) · [More workflows](workflows/) · [RAM/ATS/UI commands and settings](docs/RAM_ATS_UI_RU.md)

## Verification status

Version **0.5.0rc1**: 241 main tests, 58 audit regression tests, and one benchmark CLI test passed locally on Linux x86_64 / Python 3.11.16 / torch 2.12.0+cpu. These cover CPU contracts, small native H3/Qwen models, UI behavior, and numerical paths. They **do not prove CUDA/FSDP behavior on V100**.

The user reported successful runs of an earlier server build on an AC922 with three V100s: `fsdp2` and `fsdp2_sequence` using vllm-FA, FA, and SDPA. We have **not tested** the new `ram_min` profile, Qwen32B, Spectrum, custom wheel, or actual GPU memory/speed on that AC922. Full pretrained H3 generation for this release is **NOT_RUN** in our environment. No speedup or OOM-free operation is claimed.

For exact results, see the [0.5.0rc1 report](reports/ram-min-2026-09-29/REPORT_RU.md), [benchmarks](BENCHMARKS.md), and [known limitations](KNOWN_LIMITATIONS.md). Spectrum is a separate experimental approximation; a model-wide ATS allocator is not implemented.

## Documentation

- [Architecture and FSDP](ARCHITECTURE.md)
- [ComfyUI compatibility and execution modes](COMPATIBILITY.md)
- [GPU selection, attention, and custom vllm wheel](docs/MULTIGPU_ATTENTION.md)
- [Memory, ATS, UI, performance, and quality](docs/RAM_ATS_UI_RU.md)
- [FP16 Safe](docs/FP16_SAFE.md)
- [Earlier release notes and detailed commands (Russian)](README_HISTORY_RU.md)

Project license: [GPL-3.0](LICENSE). Third-party notices are retained in [licenses](licenses/).
