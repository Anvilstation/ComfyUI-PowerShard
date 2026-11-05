# ComfyUI-PowerShard — Wan, LTX-2, MiniMax H3 and image models (SDXL, Flux, Z-Image, Qwen-Image, MOSS) on multiple GPUs

[![CI and release](https://github.com/Anvilstation/ComfyUI-PowerShard/actions/workflows/ci-release.yml/badge.svg?branch=main)](https://github.com/Anvilstation/ComfyUI-PowerShard/actions/workflows/ci-release.yml)

[Русская версия](README_RU.md) · [Download latest release](https://github.com/Anvilstation/ComfyUI-PowerShard/releases/latest) · [Changelog](CHANGELOG.md) · [Release guide](docs/RELEASING.md)

> **Please read before installing**
>
> * **This project is entirely AI-written ("vibe-coded").** The code, tests and documentation were generated
>   by language models from the author's instructions. A human set the tasks and ran and checked the results
>   on his own server, but nobody reviewed the code line by line. Use it at your own risk.
> * **It was built for the IBM Power System AC922**: POWER9 ppc64le, 6 × NVIDIA Tesla V100 16 GB with NVLink,
>   Ubuntu 20.04, and custom builds of PyTorch 2.12, CUDA 12.4 and Triton 3.6.
> * **Real-hardware testing was done on that server only.** The package has **not been tested** on x86_64
>   servers or on other GPUs (A100/H100, RTX 30/40/50). It may work there: nothing in the code is tied to
>   POWER9, and the sm80+ code paths are written (bf16, int8/fp8/nvfp4 kernels, Triton INT8). They are
>   unverified.
> * **Image-model support, MiniMax H3 LoRA, H3 PDD checkpoints and H3 Context Windows are new in this version
>   and have not run on real hardware yet.** They were written without a GPU and checked only by tests that
>   need no CUDA. Run the tests in section 10 before relying on them.
> * It is distributed under **GPL-3.0-or-later** with no warranty of any kind. See [LICENSE](LICENSE) for the
>   license and [NOTICE](NOTICE) for third-party components. Model weights are not included.

PowerShard runs one large video or image model **on several GPUs at once**. FSDP2 splits the weights across the
cards, and sequence parallelism (token or Ulysses) splits the long sequence of video tokens. Each card holds
only its own part, so 14–22B models and long videos fit into 6 × 16 GB.

| What | Models |
|---|---|
| **Wan** | 2.2 A14B T2V / I2V (MoE high+low), TI2V-5B, S2V, Animate, WanDancer; 2.1 T2V / I2V / FLF, VACE, HuMo, SCAIL, Animate2, InfiniteTalk / MultiTalk, Fun Camera / Control, Uni3C |
| **LTX** | LTX-2 / 2.3 / 2.5 video+audio (LTXAV) and video-only (LTX-Video) |
| **MiniMax H3** | T2VA / FL2VA / R2VA; quantized weights; **Veda Sparse Attention** |
| **Image** | any model your ComfyUI loads: SDXL / SD 1.5, Flux.1 / Kontext / Krea, Flux.2, Chroma, Z-Image, Lumina 2, Qwen-Image (Edit), MOSS-Image, HiDream, SD3.5, Fibo 2 and more. Sequence parallel for Flux / Flux.2 / Chroma, Qwen-Image, Z-Image / Lumina 2 and MOSS-Image |
| **Text encoders** | umT5-XXL (Wan), Gemma (LTX), Qwen3-VL 32B (H3), T5 / CLIP (Flux), Qwen2.5-VL (Qwen-Image), Qwen3-4B (Z-Image), Qwen3.5-9B (MOSS), Mistral (Flux.2), also sharded across the GPUs |
| **Accelerators** | Block Cache (TeaCache/FBCache), NAG, RIFLEx. EasyCache/LazyCache, Context Windows and STG also work |

---

## 1. Installation

1. Copy the folder to `ComfyUI/custom_nodes/ComfyUI-PowerShard` (keep only one copy) and restart ComfyUI.
2. There are no new dependencies. The package uses your existing torch, NCCL, flash-attn and Triton. You need
   a recent ComfyUI that supports the model you want (Wan 2.2, LTX-2, MiniMax H3). Models and text encoders
   are built by your ComfyUI's own code, so newer families need a newer ComfyUI: MOSS-Image v0.42+, H3 PDD
   checkpoints v0.42+, Flux.3 v0.41+, Fibo 2 v0.43+. To check the environment, run `python scripts/diagnose.py`.
3. Put the models into the standard ComfyUI folders:

| Files | Folder |
|---|---|
| Wan (`wan2.2_*_14B_*`, `wan2.2_ti2v_5B_*`, Wan 2.1), MiniMax H3, image models (Flux, Z-Image, Qwen-Image, MOSS...) | `models/diffusion_models` |
| SDXL / SD 1.5 and other full checkpoints | `models/checkpoints` |
| LTX: full checkpoint (`ltx-2.3-22b-dev.safetensors`) or transformer only | `models/checkpoints` or `models/diffusion_models` |
| umT5-XXL, Gemma 3 12B / Gemma 4, Qwen3-VL 32B (H3) | `models/text_encoders` |
| Wan / H3 VAE | `models/vae` |
| LoRA | `models/loras` |
| Uni3C, InfiniteTalk | `models/model_patches` |
| Veda predictor | `models/veda` |

Any file format that ComfyUI reads works: bf16 / fp16 / fp32, fp8 (scaled and `comfy_quant`), int8, int4
(ConvRot / asym / w6a8), nvfp4, mxfp8. GGUF is not supported.

## 2. Example workflows

The workflows are in `example_workflows/`, in UI format. Drag a file into the ComfyUI window to open it;
recent ComfyUI versions also list them in the templates menu. The model file names are examples, so pick
your own files.

| File | What it does |
|---|---|
| `wan22_t2v_a14b.json` | Wan 2.2 A14B text-to-video, 832×480, 81 frames; both experts in one worker pool, one KSampler |
| `wan22_i2v_a14b.json` | the same from an image (`WanImageToVideo`) |
| `wan22_ti2v_5b.json` | Wan 2.2 TI2V 5B, 1280×704, 121 frames; text-to-video when no image is connected |
| `ltx2_t2v_audio.json` | LTX-2.3 video with audio from text |
| `ltx2_i2v_audio.json` | LTX-2.3 video with audio from an image |
| `minimax_h3_t2va.json` | MiniMax H3 video+audio; int8 weights stay packed in VRAM |
| `minimax_h3_t2va_veda.json` | the same plus Veda Sparse Attention |
| `sdxl_t2i.json` | SDXL 1024×1024: UNet on the workers (batch parallel), CLIP and VAE from the same checkpoint |
| `flux1_dev_t2i.json` | Flux.1 dev: image tokens split across GPUs, clip_l + t5xxl on the workers |
| `z_image_turbo_t2i.json` | Z-Image Turbo, 8 steps |
| `moss_image_t2i.json` | MOSS-Image, 50 steps, cfg 3 (Flux.2 latents) |
| `qwen_image_t2i.json` | Qwen-Image 20B, 1328×1328 |

All graphs use `sequence_mode=ulysses`, `gpu_ids=all` and the `PowerShard: освободить GPU перед VAE`
(release GPU before VAE) node. To rebuild them, run `python scripts/build_example_workflows.py`.

## 3. Nodes

The node display names are in Russian; class names are given where useful. The minimum chain is
**`PowerShard: GPU / память / attention` (PowerShardConfig) → loader → regular ComfyUI nodes**. Add the rest
only when you need it.

### Common

| Node | Purpose |
|---|---|
| `PowerShard: GPU / память / attention` (PowerShardConfig) | which GPUs to use (`all`, or `5,2,0` where the order sets the ranks); where the weights live (`gpu` = VRAM, `cpu` = pinned RAM, `ats` = experimental); attention backend; `token` / `ulysses` |
| `PowerShard: дополнительные настройки` (PowerShardConfigTuning) | VRAM reserve, prefetch, NUMA, strict attention, `allow_host_wrappers`, fp16 exchange. Usually not needed |
| `PowerShard: формат весов / вычисление` (PowerShardWeights) | `weight_format`, `compute`, `weight_dtype`, `fp16_accumulation` (section 5). Connect it to the `weights` input of any loader. When it is not connected, weights are fp16/bf16, as in plain ComfyUI |
| `PowerShard: LoRA` (PowerShardLoRA) | LoRA / LoHa / LoKr for any loader (Wan, LTX, H3, image); chain several (section 6) |
| `PowerShard: Text Encoder` (PowerShardTextEncoder) | `type`: `wan_umt5`, `ltx_gemma`, `flux`, `flux2`, `qwen_image`, `z_image`, `moss_image`, `sdxl`, `sd3`, `hidream` …; `placement`: `workers` (on the GPUs) or `cpu_fp32` (CPU, FP32) |
| `PowerShard: освободить GPU перед VAE` (PowerShardRelease) | parks or closes the workers before decoding |
| `PowerShard: Free VRAM` (PowerShardFreeVRAM) | unloads native ComfyUI models (CLIP vision, VAE) from cuda:0 before sampling |
| `PowerShard: checkpoint info (Wan / LTX / image)` (PowerShardInfo) | family, geometry, format and memory estimate without loading the weights |
| `PowerShard: диагностика среды` (PowerShardDiagnostics) | GPUs, torch, NCCL, attention providers |

Every loader has **`keep_in_memory`**:

* `true` (the default): between jobs the weights stay in worker RAM as shards and VRAM is freed, so the next
  job starts without reading the file again.
* `false`: the workers close after every job.

### Wan

| Node | Purpose |
|---|---|
| `PowerShard Wan 2.2: MoE Loader (high+low)` | both A14B experts in one worker pool. The `moe` output feeds a single sampler and switches experts by itself (`boundary=0` means auto: T2V 0.875, I2V 0.900). The `high_noise` / `low_noise` outputs feed two `KSamplerAdvanced` nodes |
| `PowerShard Wan: Sequence Loader (одна модель)` | TI2V-5B, Wan 2.1 and its variants, or a single expert |
| `PowerShard Wan: численная политика / память` (PowerShardWanOptions) | `fp16_safe`, `mlp_chunk_mode` / `mlp_chunk_tokens`, `moe_residency` (`swap` / `both`), `batch_chunk`, `model_type` |
| `PowerShard Wan: Uni3C ControlNet`, `… InfiniteTalk / MultiTalk`, `… Animate2 Cache` | special modes |

The model variant is detected from the file header. Native nodes build the conditioning: `WanImageToVideo`,
`Wan22ImageToVideoLatent`, `WanVaceToVideo`, `WanSoundImageToVideo`, `WanAnimateToVideo`,
`WanHuMoImageToVideo`, `WanSCAILToVideo`, `WanDancerVideo`, `WanAnimate2ToVideo`, `WanCameraImageToVideo`,
`WanFunControlToVideo`.

`moe_residency` controls where the two experts live:

* `swap` (the default): one expert in VRAM, the other in RAM.
* `both`: both experts in VRAM. With `gpu` placement on V100 that is about 9.5 GB of weights per card, which
  leaves little room for activations.

### LTX-2

| Node | Purpose |
|---|---|
| `PowerShard LTX-2/2.5: Sequence Loader` | full checkpoint or transformer only |
| `PowerShard LTX: численная политика / память` (PowerShardLTXOptions) | `fp16_safe`, FFN chunking, `batch_chunk`, `attention_chunk` |
| `PowerShard LTX: Video+Audio VAE из checkpoint` | video and audio VAE plus vocoder, without reading the transformer weights |

Video tokens are split across the GPUs. Audio tokens (a few hundred) are replicated on each GPU.
Audio→video attention uses an exact partial softmax plus all-reduce.

The loader works with the native LTXV nodes: Context Windows, EasyCache / LazyCache, STG, Modality Guidance,
DualCFGGuider, guides / keyframes / IC-LoRA and the latent upscaler.

On 6 × V100:

* use `weight_placement=gpu` (22B fp16 is about 7.5 GB per GPU) up to about 121 frames at 768×512;
* for longer videos, use `cpu` with `mlp_chunk_mode=auto`.

### MiniMax H3

| Node | Purpose |
|---|---|
| `PowerShard MiniMax H3: Loader` (PowerShardH3QuantLoader) | H3 with any weight format (via the `weights` input), LoRA (via the `lora` input: PTD, FastH3, PDD LoRAs and others), PDD checkpoints, Context Windows / EasyCache, Cancel inside a step, Veda |
| `PowerShard MiniMax H3: Veda Sparse Attention` (PowerShardH3Veda) | sparse attention (see below) |
| `PowerShard H3 Qwen Loader` | the Qwen3-VL 32B text encoder of H3 in a worker pool |
| `PowerShard: H3 Sequence Loader`, `… FP16 Patcher`, `… H3 MLP chunk`, `… Qwen MLP chunk`, `PowerShard Spectrum` | the original PowerShard 0.5.3 path, unchanged |

New for H3 in this version, following ComfyUI v0.41–0.43:

* **LoRA:** the keys are matched by ComfyUI's own `comfy.lora`, including single-head LoRAs for PDD head
  banks; the delta is merged into each GPU's local rows. It cannot be merged into native INT8
  (`precision=int8_fp16`): choose `fp16` and the `int8` format in the weights node instead.
* **PDD checkpoints** (`pdd_steps` in the metadata), which stack output heads. Needs ComfyUI v0.42+.
* **Context Windows and EasyCache / LazyCache** run on the host, as with Wan/LTX.
* The native `MiniMax H3 Latent Upscale With Model`, LightVAE and Sol-H3-VAE work on the host unchanged.

#### Veda Sparse Attention

[Veda](https://github.com/veda-sparse/Veda-on-ComfyUI) (ICML 2026) uses a small trained predictor to score
3D tiles of 128 tokens. Video×video attention is then computed only for the best tiles: by default 32 key
tiles per query tile, about 10%. Text and audio stay dense, and the H3 weights are not changed.

To set it up:

1. Put the predictor (about 275 MB) into `models/veda/`. There are two:
   [T2VA](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) and
   [R2VA](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview).
2. In `PowerShard: GPU / память / attention`, set **`sequence_mode=ulysses`**. This is required with more
   than one GPU.
3. Chain `PowerShard MiniMax H3: Loader` → `PowerShard MiniMax H3: Veda Sparse Attention` → sampler.

| Input | Default | Meaning |
|---|---|---|
| `generated_sparsity` | `32` | key tiles kept per query tile. `90%` gives the skipped fraction instead; `trained` uses the training value |
| `reference_sparsity` | `32` | the same for references (first/last frame, reference). `0%` means full attention |
| `full_attention_layers` / `full_attention_steps` | empty | blocks or steps that use full attention, e.g. `0, 1, 47-49` |
| `backend` | `auto` | see below |

`backend` options:

* `auto`: `torch-sdpa` on V100; on sm80+, the fastest backend that passed the check.
* `triton-int8`: sm80+.
* `triton-fp16`: Triton without tensor cores on V100. Slow; kept for comparison.
* `torch-sdpa`: no Triton.

Changing the Veda settings does not restart the workers. The first step also loads the predictor and
compiles the kernels.

On V100, Triton 3.x does not use the Volta tensor cores. `triton-fp16` is therefore slow there: the GPU shows
100% load but draws only about 150 W of its 300 W. Use `auto` or `torch-sdpa` instead.

### Image models

| Node | Purpose |
|---|---|
| `PowerShard Image: Loader` (PowerShardImageLoader) | any ComfyUI image model (`diffusion_models/`, or a full checkpoint from `checkpoints/`); inputs `parallel`, `weights`, `lora`, `keep_in_memory` |
| `PowerShard Image: CLIP + VAE из checkpoint` (PowerShardCheckpointCLIPVAE) | CLIP and VAE of a full checkpoint (SDXL, SD 1.5) on the host, without loading the UNet |

ComfyUI itself builds the model (family detection, `model_sampling` and conditioning are native) and the
forward is the model's native code. PowerShard shards the weights across the GPUs (FSDP2) and splits the
compute:

| `parallel` | How | Use it for |
|---|---|---|
| `sequence` | image tokens are split across GPUs; attention keys/values are gathered from all of them | Flux / Flux.2 / Chroma, Qwen-Image, Z-Image / Lumina 2, MOSS-Image; speeds up even batch 1 |
| `batch` | batch rows (cond / uncond, batch_size) are split across GPUs | any model, including SDXL; with CFG it uses 2 GPUs, with batch_size 3 it uses 6 |
| `replicate` | every GPU computes the same thing | memory only (the model does not fit one card) |
| `auto` | `sequence` when the family supports it, otherwise `batch` | the default |

Dtypes (`weight_dtype` in the weights node):

* `auto` gives bf16 on sm80+, and fp16 on V100 when the model supports it (Flux, Chroma, SDXL).
* bf16-only models (Z-Image, Qwen-Image, MOSS-Image, HiDream) on V100 get bf16 storage with exact FP32 compute.
  That is slower; `weight_dtype=fp16` is faster, but check the quality.
* Quantized files (fp8 / int8 / nvfp4 / int4) and load-time quantization work as for Wan/LTX.

Works: native ControlNets (computed on the host and passed to the workers), EasyCache / LazyCache, CFG nodes
(CFGNorm, APG and similar), `ModelSampling*` shifts, Flux Kontext / Qwen-Image Edit / MOSS-Image Edit
references, and Skip Layer Guidance (`SkipLayerGuidanceDiT` / `SkipLayerGuidanceDiTSimple`): the list of skipped
blocks is sent to the workers. SLG needs a model whose native forward supports block replacement (Flux, Chroma,
Qwen-Image, MOSS-Image); Lumina 2 / Z-Image do not have it.

Does not work: IP-Adapter, PuLID and other attention/block patches (`patches`, other `patches_replace`), the native
`LoraLoader` (use `PowerShard: LoRA`), and `CLIPSetLastLayer` with the distributed encoder (for SD 1.5 with
clip skip, use the native `CLIPLoader`).

### Accelerators

| Node | What it does |
|---|---|
| `PowerShard: Block Cache` | TeaCache/FBCache: blocks are skipped when the residual of block 0 changed less than `threshold` (0.05–0.12). All GPUs make the same decision. `max_cache_mb` caps the cache VRAM per GPU |
| `PowerShard: NAG` | negative prompt without CFG (distilled / Lightning at cfg=1) |
| `PowerShard Wan: RIFLEx` | videos longer than 81 frames without repeating motion |

Third-party `forward` patches are not carried into the workers. This includes community TeaCache/MagCache
and the KJ SageAttention patch. Using one gives a clear error.

## 4. Text encoders

With `placement=workers`, `PowerShard: Text Encoder` keeps the encoder on the same GPUs as the generator and
computes in FP32. This keeps the encoder's 11–24 GB off cuda:0.

`after_encode` controls what happens to the encoder workers:

* `release`: they close when the generator starts.
* `release_now`: they close right after encoding.
* `keep_ram`: the workers stay alive between generations. When the generator starts, the weights are parked in RAM;
  they return to the GPUs later without reading the file again.

Repeated prompts come from a cache. For `ltx_gemma`, choose `ltx_checkpoint` (the LTX file with the text
projection). For encoders made of several files (`flux` = clip_l + t5xxl, `sdxl` = clip_l + clip_g, `sd3`,
`hidream`), put the other files into `text_encoder_2…4`. Small CLIPs (SDXL) are easier left to the native
`CLIPLoader`.

`placement=cpu_fp32` runs the native ComfyUI encoder on the CPU in FP32. It is slower but does not touch the
GPUs.

## 5. Weight format and compute

These are the options of the `PowerShard: формат весов / вычисление` node:

| Option | Values |
|---|---|
| `weight_format` | `dequantize`: fp16/bf16 (the default). `as_file`: as stored in the file (int8/int4/nvfp4/fp8 stay quantized). `int8` / `fp8` / `mxfp8` / `nvfp4` / `int4`: quantize at load time |
| `compute` | `auto`: native kernels when the GPU and comfy-kitchen support the format, otherwise dequantization. `native`. `dequantize`: the weight is converted to fp16/bf16 for the GEMM; works everywhere |
| `weight_dtype` | `auto` means bf16 on sm80+ and fp16 on V100/T4 (image models without fp16 on V100: bf16 + FP32 compute); or `fp16`, `bf16`, `fp32` (image only). H3 is always fp16 |
| `fp16_accumulation` | cuBLAS accumulates fp16 GEMMs in fp16. Up to 2× faster, but only on RTX 30/40/50 with fp16 weights. No speed-up on V100/A100/H100; lower precision everywhere |

Native kernel requirements:

| Format | Requires |
|---|---|
| int8 / int4 | sm75+ and the comfy-kitchen CUDA backend (torch cu130+) |
| fp8 | sm89+ |
| nvfp4 / mxfp8 | sm100+ |

The AC922 has CUDA 12.4, so it always dequantizes.

Recommended settings:

* **6 × V100:** `dequantize`, which is the fastest. If VRAM runs out, use `as_file` or `int8` with
  `compute=dequantize`.
* **RTX 5090:** `as_file`, or `nvfp4` / `fp8`, with `compute=auto`.

## 6. LoRA

One `PowerShard: LoRA` node serves every loader (type `POWERSHARD_LORA`; chain it for several LoRAs;
`apply_to` only matters for the Wan 2.2 MoE experts). The old `PowerShard Wan/LTX: LoRA` nodes are hidden from
search but still open in old workflows.

LoRA is merged at load time directly into each GPU's local rows (`W[a:b] += s · up[a:b] @ down`). The full
weight is never assembled, and changing a LoRA reloads the workers.

* **Wan / LTX:** kohya, diffusers, PEFT, mochi, LoHa, LoKr and `.diff` / `.diff_b`; keys that do not match the
  model are skipped with a warning.
* **MiniMax H3 and image models:** ComfyUI's own `comfy.lora` matches the keys (every format ComfyUI accepts for
  that model, including Flux fused qkv and H3 PDD banks) and its adapters compute the delta.
* **Rejected:** DoRA, OFT, BOFT, GLoRA, LoCon-mid and Tucker, which need the full weight.
* **Native loader:** `LoraLoaderModelOnly` on a PowerShard model gives an error with a hint.

## 7. Cancel and VRAM

* **Cancel during a step:** the Wan, LTX, image and `PowerShard MiniMax H3: Loader` workers check for cancellation
  before every DiT block. All GPUs leave on the same block, without waiting for the step to finish. The
  original H3 Sequence Loader waits for the end of the step.
* **A second Cancel while draining** forces a reset: the workers stop and VRAM is freed.
* **ComfyUI's "Unload models" / "Free model and node cache"** stop all PowerShard workers. So does
  `--disable-smart-memory`.
* **A hung rank:** draining has a time limit and NCCL operations have a timeout (see the variables below).

`nvidia-smi` may show used memory that belongs to no process. This is how the driver reports NCCL buffers
(cuMem) and `expandable_segments` allocations of the workers themselves. If there are no orphaned processes
(`ps -ef | grep wan_worker`), the memory is freed together with the workers.

## 8. Memory and speed

* **First job:** starts the workers, checks attention and reads the checkpoint. Later jobs with
  `keep_in_memory=true` do not read the file again. Changing the config, LoRA or options means a new load.
* **`mlp_chunk_mode=auto`:** splits the FFN by tokens according to free VRAM. Recommended for 720p on 16 GB.
* **Ulysses vs. token:** Ulysses is recommended for Wan 14B (40 heads → 7 per GPU on 6 GPUs) and for long
  videos. `token` gathers the full K/V on every GPU.
* **`weight_placement=ats`:** experimental (POWER9 + NVLink ATS). On V100, use `gpu` or `cpu`.
* **cuda:0:** native ComfyUI models (VAE, CLIP vision) live on cuda:0, which is also rank 0. Use the worker
  text encoders and `Free VRAM`.

## 9. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `POWERSHARD_DRAIN_TIMEOUT_S` | auto (≥ 1200) | how long to wait for the step to finish after Cancel |
| `POWERSHARD_NCCL_TIMEOUT_MIN` | 60 | timeout of a single NCCL operation in the workers |
| `POWERSHARD_SHUTDOWN_GRACE_S` | 3 | how long to wait for a clean worker exit before SIGTERM |
| `POWERSHARD_KEEP_REPORTS` | 300 | how many report files to keep in `output/powershard/` |
| `POWERSHARD_CANCEL_CHECK_BLOCKS` | 1 | check for Cancel every N blocks |
| `POWERSHARD_DEBUG_FINITE` | 0 | `1` checks every block for NaN/Inf (diagnostics, slow) |
| `POWERSHARD_PROFILE=1` | off | torch.profiler trace of the first forward passes |

Rank logs are in `<ComfyUI>/output/powershard/`: `*-rankN.log`, `*-progress.json`, `run-*.json`.

## 10. Testing on your own server

Run these from the custom node folder with your ComfyUI's Python:

```bash
# no GPU: nodes, workflows, formats, LoRA, worker lifecycle, Veda settings
python -m pytest tests/test_wan_config.py tests/test_ltx_config.py tests/test_lifecycle_wan.py \
                 tests/test_veda_config.py tests/test_image_config.py tests/test_example_workflows.py
# image: sequence / batch parallel against a plain forward (CPU, gloo, native Flux and Qwen-Image)
python -m pytest tests/test_image_parallel_torch.py --comfy /path/to/ComfyUI
# CPU + torch: equivalence with the native ComfyUI models
python -m pytest tests/test_wan_native.py tests/test_wan_pipeline.py tests/test_ltx_native.py \
                 tests/test_quant_formats_torch.py --comfy /path/to/ComfyUI
python -m pytest tests/test_lora_torch.py tests/test_veda_torch.py
# end-to-end CUDA/NCCL checks
python scripts/probe_wan_cuda.py --comfy /path/to/ComfyUI --gpus all --sequence-mode ulysses --moe
python scripts/probe_ltx_cuda.py --comfy /path/to/ComfyUI --gpus all --sequence-mode ulysses
python scripts/probe_veda_cuda.py --gpu 0 --grid 37x24x42 --heads 7 --predictor /path/to/models/veda/<file>.safetensors
```

## 11. Limitations

* **Hardware:** tested only on AC922 with 6 × V100 (see the top of this page). Other GPUs and x86_64 are
  untested.
* **Not supported:** GGUF, torch.compile, third-party `patches_replace`, Spectrum for Wan/LTX, in-context
  latents.
* **Veda:** MiniMax H3 only, and only through `PowerShard MiniMax H3: Loader`. Ulysses is required on
  multiple GPUs.
* **Image:** sequence parallel only for the families listed above. In `sequence` mode attention uses
  ComfyUI's global backend (a per-layer `comfy_attention` config in the model file is not used). Qwen-Image 2.1
  runs without its prefix cache (batch / replicate). IP-Adapter / PuLID / block patches are not carried into the
  workers.
* **Block Cache:** a heuristic. If you see artifacts, lower `threshold` or `max_consecutive_skips`.
* **Animate2:** no Block Cache, NAG or RIFLEx. InfiniteTalk and Uni3C work with Wan T2V/I2V only.
* **Memory figures:** estimates, not guarantees.

## 12. License and provenance

* **Package code:** **GPL-3.0-or-later** ([LICENSE](LICENSE)). This is a modified version of
  ComfyUI-PowerShard 0.5.3. The 0.5.3 files are unchanged; everything new is in separate files and in lines
  appended to `__init__.py`. The full diff is `review/changes_wan22.patch`.
* **Third-party code and ideas:** listed in [NOTICE](NOTICE) and `licenses/`:
  * ComfyUI (GPL-3.0);
  * Veda-on-ComfyUI and Miowtion (MIT);
  * SageAttention (BSD-3 / Apache-2.0);
  * Comfy Kitchen and Raylight (Apache-2.0);
  * Spectrum (GPL-3.0);
  * the MIT-licensed H3 FP16-fix projects.
* **Model weights:** not included; each model is used under its own license:

  | Model | License |
  |---|---|
  | Wan | Apache-2.0 |
  | LTX-2 | LTX-2 Community License |
  | MiniMax H3 and the Veda predictors | MiniMax H3 Community License |
  | Gemma | Gemma Terms of Use |
  | Qwen3-VL, umT5 | Apache-2.0 |
  | SDXL | CreativeML Open RAIL++-M |
  | Flux.1 dev, Flux.2 dev | FLUX Non-Commercial License |
  | Z-Image, Qwen-Image, MOSS-Image | Apache-2.0 |

  Accepting those terms is the user's responsibility.
* **Trademarks:** ComfyUI, Wan, LTX, MiniMax, NVIDIA and IBM are trademarks of their owners. This project is
  not affiliated with them.

## 13. Files

| Path | Contents |
|---|---|
| `powershard/wan_*.py`, `ltx_*.py` | Wan and LTX: headers, model, backend, worker, ComfyUI adapter, nodes |
| `powershard/common_nodes.py` | weight format, LoRA, text encoder, checkpoint info |
| `powershard/image_*.py`, `comfy_lora.py` | image models: host proxy, worker, sequence/batch parallel, LoRA through comfy.lora |
| `powershard/native_te.py`, `wan_text.py` | text encoders in workers |
| `powershard/h3_quant.py`, `veda/`, `veda_nodes.py` | MiniMax H3 loader and Veda |
| `powershard/quant_*.py` | weight formats, quantized Linear layers in FSDP |
| `powershard/accel.py` | Block Cache, NAG, RIFLEx |
| `powershard/lifecycle.py`, `cancel.py` | Cancel, draining, worker shutdown |
| rest of `powershard/` | the original PowerShard 0.5.3 (H3, Qwen, config, attention, FSDP) |
| `example_workflows/` | example graphs; `scripts/build_example_workflows.py` generates them |
| `scripts/`, `tests/` | CUDA probes, diagnostics, tests |
| `docs/history/` | historical reports for 0.2–0.5.3 (Russian) |
