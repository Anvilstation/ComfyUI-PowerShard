# Wan 2.2 в ComfyUI на IBM AC922 (6×V100 16 GB): ресерч

Дата: 2026-10-07. Целевой стенд: POWER9 ppc64le, 6×Tesla V100 16 GB (sm_70), 128 GB RAM, Ubuntu 20.04.5, PyTorch 2.12 / CUDA 12.4, собранный FA2 для sm_70. Цифры производительности ниже — **оценки**, не измерения на этом сервере.

## 1. Что такое Wan 2.2 (то, что важно для железа)

| Модель | Архитектура | Параметры | FP16 вес | Латент / VAE | Особенности |
|---|---|---|---|---|---|
| T2V-A14B | 2 эксперта (MoE по шуму): high-noise + low-noise, каждый = Wan 2.1-14B (dim 5120, 40 слоёв, 40 heads×128, FFN 13824) | 2×14 B (активен один) | 2×≈28.6 GB | 16 каналов, Wan 2.1 VAE (8×8×4) | boundary 0.875, shift 12, 40 шагов, CFG (3.0, 4.0) |
| I2V-A14B | то же, in_dim 36 (латент + маска + картинка), без CLIP-vision | 2×14 B | 2×≈28.6 GB | Wan 2.1 VAE | boundary 0.900, shift 5, CFG 3.5 |
| TI2V-5B | один DiT (dim 3072, 30 слоёв, 24 heads), in/out 48 | 5 B | ≈10 GB | Wan 2.2 VAE (16×16×4) | per-frame timesteps при I2V (denoise_mask) |

Эксперты переключаются по timestep: high-noise работает, пока `t ≥ boundary·1000` (t = σ·1000 у flow-моделей ComfyUI). В ComfyUI это обычно два `KSamplerAdvanced` (high: шаги 0…k с `return_with_leftover_noise`, low: k…конец), shift 8 через `ModelSamplingSD3`.

Ускорение шагов: LoRA Wan2.2-Lightning (lightx2v) — 4 шага вместо 40, CFG 1 (раздельные файлы high/low). Для 6×V100 это самый большой выигрыш по времени: сокращение числа forward-ов в ~10–20 раз, тогда как любые улучшения kernels дают проценты.

ComfyUI официально рекомендует Wan в **fp16** (ближе к fp32-оригиналу, чем bf16), что удачно для V100 без bf16 tensor cores.

## 2. Ограничения стенда

* **14B в одну V100 не помещается**: 28.6 GB fp16 > 16 GB; fp8 (14.3 GB) тоже не оставляет места под активации, а V100 не умеет fp8-математику. Значит нужен **шардинг весов (FSDP)** или offload.
* **bf16 на V100 нет в tensor cores** → считать в fp16; bf16/fp8 чекпоинты надо конвертировать в fp16 при загрузке.
* **FlashAttention-2 официально требует sm80+**; для sm_70 есть только кастомные порты (у вас собран). Fallback — SDPA (mem-efficient cutlass kernel работает на Volta) или tiled math.
* **ppc64le**: нет колёс Ray/xFuser/yunchang и т.п.; всё стороннее приходится собирать. Решения на Ray (Raylight) здесь трудно развернуть.
* **AC922 плюсы**: NVLink 2.0 не только GPU↔GPU, но и CPU↔GPU (≈50–75 GB/s на GPU в 6-GPU конфигурации против ~12 GB/s PCIe3), аппаратный ATS. Поэтому хранение весов в RAM (CPUOffloadPolicy/ATS) с подкачкой активного блока здесь дешевле, чем на x86+PCIe.

## 3. Варианты запуска Wan 2.2 в ComfyUI

| Подход | Суть | Для 6×V100 ppc64le |
|---|---|---|
| Native ComfyUI, 1 GPU + `--lowvram`/block swap | веса в RAM, по блоку на одну V100 | работает, но 5 карт простаивают; PCIe/NVLink-подкачка 28 GB на каждый шаг ×2 CFG; на 720p очень долго |
| ComfyUI-MultiGPU (DisTorch) | раскладывает слои/кэш по разным GPU, вычисление последовательное | экономит память, не ускоряет: в каждый момент работает одна карта |
| Raylight (Ray + xDiT/xFuser, USP + FSDP + CFG-parallel) | ближе всего к цели, поддерживает Wan 2.2, Volta указан | нужны Ray, xFuser, yunchang, новый NCCL под ppc64le — отдельный тяжёлый porting-проект; заменяет ваш стек |
| xDiT/xFuser напрямую | Ulysses/Ring sequence parallel | USP без FSDP требует полную копию весов на каждой GPU → 14B не помещается |
| **PowerShard (ваш) + Wan-ноды (этот пакет)** | FSDP2 Shard(0) весов + sequence parallel (token/Ulysses) + RAM/ATS placement, без Ray | использует уже работающий стек (NCCL, custom FA2, probes, RAM-парковка); добавлен только Wan |

Вывод: правильная схема для этой машины — **FSDP2 (шарды весов на 6 GPU или в RAM) + sequence parallel по токенам видео**, все 6 карт считают одновременно свою часть последовательности. Это ровно архитектура PowerShard для MiniMax H3; её и расширяем.

## 4. Бюджет памяти (на одну V100)

Токены: 480p/81 кадр → латент 21×60×104 → 21·30·52 = **32 760** токенов; 720p/81 → **75 600**.

| Компонент | 480p, 6 GPU | 720p, 6 GPU |
|---|---|---|
| Шард одного эксперта A14B fp16 | 4.8 GB | 4.8 GB |
| Оба эксперта резидентно (`moe_residency=both`, placement=gpu) | 9.5 GB | 9.5 GB |
| Собранный FSDP блок (+prefetch) | 0.7 GB (×2 при prefetch=1) | то же |
| Локальная последовательность (B=2 CFG) | 10 920 строк ×5120 FP32 ≈ 0.22 GB на тензор | 25 200 строк ≈ 0.52 GB |
| Ulysses обмен q/k/v (FP32 wire) | ~1.4 GB send+recv | ~3.2 GB; с `batch_chunk=1` вдвое меньше |
| CUDA context + NCCL | ~0.6–0.8 GB | то же |

Рекомендации (заложены в `workflows_wan/`):

* **480p, A14B**: `placement=gpu`, `moe_residency=swap` (в VRAM один эксперт; переключение high→low — один перенос ~4.8 GB на GPU по NVLink за прогон), Ulysses, FP32 wire, MLP auto.
* **720p, A14B**: `placement=cpu` (оба эксперта в pinned RAM ≈ 57 GB), `batch_chunk=1`, MLP auto, `prefetch_blocks=1`. Возможен `sequence_comm_dtype=fp16` для экономии обмена.
* **TI2V-5B**: веса 1.7 GB/GPU, `placement=gpu`, 720p (1280×704, 121 кадр ≈ 27 280 токенов) помещается свободно.
* **umT5-XXL**: на CPU в FP32 (≈23 GB RAM) — не конкурирует с ранками за VRAM и без fp16-переполнений.
* **VAE**: `PowerShardRelease` перед `VAEDecodeTiled` (освобождает VRAM ранков; веса остаются в RAM).

## 5. Порядок величины по времени (оценка)

FLOPs одного forward A14B на 480p, B=2: ≈1.8·10¹⁵ (attention) + ≈1.8·10¹⁵ (GEMM) ≈ 3.6 PFLOP. При реальных ~40–60 TFLOPs fp16 на V100 и 6 картах — ~10–15 с/шаг; 20 шагов с CFG — ~4–5 мин, с Lightning (4 шага, CFG 1, B=1) — ~0.5 мин плюс VAE. 720p в ~3–4 раза дороже из-за квадратичного attention. Обмен (Ulysses ~1 GB/rank/блок·40 и all-gather весов 28 GB/rank/forward) по NVLink меньше вычислений. Реальные числа — только после `scripts/probe_wan_cuda.py` и замера на ваших чекпоинтах.

## 6. Что реализовано в Wan-нодах (кратко)

* Один пул воркеров на **оба эксперта** (два FSDP root в одной NCCL group). Выходы `high_noise`, `low_noise`, `moe` — clones одной модели, ComfyUI не выгружает «соседа». После high-прохода workers не паркуются и не закрываются — ждут low-проход.
* Sequence parallel: token (all-gather K/V) и **Ulysses** (all-to-all по heads; 40 heads дополняются нулями до 42 на 6 GPU, 24 — без паддинга). Cross-attention к тексту локальна. Собирается только выход head (64/192 канала), а не hidden 5120.
* Численная схема: residual, модуляция AdaLN, LayerNorm/RMSNorm, RoPE, time embedding — FP32 (как `autocast(float32)` в оригинальном Wan); GEMM и attention — FP16; q/k/v масштабируются степенями двойки до FP16 (границы из весов RMSNorm), опция Safe GEMM из H3.
* Чекпоинты: fp16, bf16, **fp8_scaled / fp8** (деквантизация локальных строк в FP16 при загрузке), native int8_tensorwise; префиксы `model.diffusion_model.`/`diffusion_model.`.
* **LoRA** (в т.ч. lightx2v): слияние в локальные строки шардов при загрузке, отдельно для high/low.
* Wan 2.2 TI2V per-frame timesteps (`Wan22ImageToVideoLatent`), Wan 2.1 I2V/FLF (CLIP image tokens), `reference_latent`, `ScaleROPE`.

## 7. Риски и что проверить на сервере первым

1. `python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus all --moe --fp8 --lora` — FSDP2/NCCL/FA2 путь на настоящих V100 против native FP32.
2. `pytest tests/test_wan_native.py tests/test_wan_pipeline.py tests/test_wan_distributed.py --comfy /opt/ComfyUI` — эквивалентность с native ComfyUI Wan на CPU, жизненный цикл MoE, gloo-обмены.
3. Попробовать `attention_backend=auto` и сравнить с `sdpa` по времени: custom FA2 на sm_70 должен пройти probe для 40 и 7 heads.
4. Пиковая VRAM по `/powershard/status` на 480p с `swap` и `both`.

## Источники

* [Wan 2.2 I2V-A14B config (boundary 0.900, shift 5.0, dim 5120, 40 layers)](https://huggingface.co/spaces/MindOfDev/Wan-2.2-5B/blob/6961549539ee2dd5be003c74bf9dc1f44217eb7f/wan/configs/wan_i2v_A14B.py)
* [NVIDIA NeMo: Wan 2.2 T2V-A14B](https://docs.nvidia.com/nemo/automodel/model-coverage/diffusion/wan-ai/wan-2-2-t2v-a14b)
* [ComfyUI-WanMoeKSampler — переключение экспертов по timestep (0.875 T2V / 0.900 I2V)](https://github.com/stduhpf/ComfyUI-WanMoeKSampler)
* [Comfy blog: Wan в fp16 ближе к fp32, чем bf16](https://blog.comfy.org/i/158757892/wan-in-fp)
* [Raylight (Ray + xDiT + FSDP для ComfyUI)](https://github.com/komikndr/raylight)
* [ComfyUI-MultiGPU](https://gitee.com/ITG/ComfyUI-MultiGPU)
* [Wan2.2-Lightning (4-step LoRA)](https://huggingface.co/lightx2v/Wan2.2-Lightning)
* [Flash Attention v2 для Pascal/Volta (сторонний порт)](https://github.com/sirCamp/flash-attention-legacy)
* [Cornell CVW: V100 + NVLink 2.0 с POWER9](https://cvw.cac.cornell.edu/gpu-architecture/gpu-example-tesla-v100/v100_mem_nvlink2)
* [TOP500: IBM POWER9 AC922](https://top500.org/news/ibm-launches-power9-servers-initial-offering-takes-aim-at-enterprise-ai/)
* Исходники ComfyUI `comfy/ldm/wan/model.py`, `comfy/model_base.py` (ревизии f1072eb и 7d9e5a0 совпадают по Wan).
