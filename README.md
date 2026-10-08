# ComfyUI-PowerShard

**Distributed MiniMax H3 and Wan inference in ComfyUI, with LTX support.** Run a single generation across selected CUDA GPUs using FSDP2 weight sharding and token or Ulysses sequence parallelism. Keep native ComfyUI conditioning, sampling, and VAE nodes in your workflow.

[Русский](README_RU.md) · [Latest release](https://github.com/Anvilstation/ComfyUI-PowerShard/releases/latest) · [Compatibility](COMPATIBILITY.md) · [License](LICENSE)

## Features

| Area | Included in this release |
|---|---|
| MiniMax H3 | FL2VA / Ref2VA, FP16 Safe, INT8 ConvRot, distributed Qwen3-VL conditioning. |
| Wan 2.1 / 2.2 | T2V, I2V, A14B high/low MoE, TI2V-5B, LoRA, specialized video/control variants. |
| LTX | LTX-Video / LTX-AV loaders, audio/video workflows, LoRA / IC-LoRA, distributed Gemma encoding. |
| GPU selection | Any nonempty set of CUDA-visible devices: `0`, `0,1,2`, `5,2,0`, or `all`. |
| Distributed execution | FSDP2 shards weights; token / Ulysses modes distribute sequence work. |
| Memory | GPU shards, CPU offload, RAM parking between runs, experimental ATS managed memory. |
| Attention and precision | Selectable attention providers, FP16 Safe for V100, configurable quantized weight storage where supported by the model and installed ComfyUI. |
| Workflows | MiniMax in [workflows](workflows/), Wan in [workflows_wan](workflows_wan/), LTX in [workflows_ltx](workflows_ltx/). |

## Installation

Clone into your existing ComfyUI installation:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Anvilstation/ComfyUI-PowerShard.git
```

Alternatively, download the release ZIP and extract its `ComfyUI-PowerShard` directory into `ComfyUI/custom_nodes`. When updating, back up the previous folder outside `custom_nodes` and replace it; keep only one installed copy. Restart ComfyUI and refresh the browser.

Use the Python environment that already runs ComfyUI. PowerShard does not install or replace PyTorch, CUDA, NCCL, or your custom attention wheels. Inspect the environment first:

```bash
cd ComfyUI-PowerShard
python scripts/diagnose.py
python scripts/check_source_requirements.py
python scripts/dependency_plan.py
```

Install missing runtime dependencies only if needed. Platform details: [x86_64](docs/INSTALL_X86_64.md), [POWER9 / ppc64le](docs/INSTALL_PPC64LE.md). Model weights are downloaded separately into standard ComfyUI model directories.

## Quick start

1. Open a workflow for your model family and select your local checkpoints, text encoder, and VAE.
2. Set `gpu_ids`, `weight_placement`, `attention_backend`, and `sequence_mode` in `PowerShardConfig`. `cpu` keeps weight shards in RAM while computation runs on GPUs.
3. Connect the PowerShard loader's `MODEL` output to the native sampler. For Wan 2.2 A14B, use the MoE loader and the supplied high/low workflow.
4. Adjust the prompt and generation settings, then queue the workflow. Supplied filenames and GPU selections are examples; replace them with your own.

Detailed model guides (Russian): [MiniMax H3 / Qwen](README_H3_RU.md), [Wan 2.1 / 2.2](README_WAN22_RU.md), [LTX / Gemma / accelerators / weight formats](README_LTX_RU.md).

## Validation and limits

Version **0.5.3** is the supplied MiniMax + Wan + LTX build. Publication checks and exact test results: [release verification](reports/release-2026-10-08/verification.json). Historical H3 results: [benchmarks](BENCHMARKS.md).

CPU tests do not certify full pretrained generation on CUDA/NCCL or POWER9. GPU performance, physical VRAM release, and ATS behavior require validation on the target machine. The Wan/LTX guides include native and CUDA probes and model-specific limitations. See [known limitations](KNOWN_LIMITATIONS.md). Cache and acceleration options may affect output quality; no measured speedup or universal memory guarantee is claimed.

Licensed under [GPL-3.0](LICENSE). Third-party notices: [NOTICE](NOTICE), [licenses](licenses/). Model licenses apply separately.
