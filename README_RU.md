# ComfyUI-PowerShard — Wan, LTX-2, MiniMax H3 и image-модели (SDXL, Flux, Z-Image, Qwen-Image, MOSS) на нескольких GPU

[![CI и релизы](https://github.com/Anvilstation/ComfyUI-PowerShard/actions/workflows/ci-release.yml/badge.svg?branch=main)](https://github.com/Anvilstation/ComfyUI-PowerShard/actions/workflows/ci-release.yml)

[English version](README.md) · [Скачать последний релиз](https://github.com/Anvilstation/ComfyUI-PowerShard/releases/latest) · [История изменений](CHANGELOG.md) · [Как выпустить релиз](docs/RELEASING.md#как-выпустить-релиз)

> **Важно прочитать до установки**
>
> * **Проект полностью написан с помощью ИИ («вайбкод»).** Код, тесты и документация сгенерированы языковыми
>   моделями по заданиям автора; человек ставил задачи, запускал и проверял результат на своём сервере, но
>   код построчно не ревьюился. Используйте на свой риск.
> * **Разрабатывался для IBM Power System AC922** (POWER9 ppc64le, 6 × NVIDIA Tesla V100 16 GB, NVLink,
>   Ubuntu 20.04, собственные сборки PyTorch 2.12 / CUDA 12.4 / Triton 3.6).
> * **Тесты на реальном железе проводились только на этом сервере.** На x86_64 серверах и других GPU
>   (A100/H100, RTX 30/40/50) пакет **не тестировался**. Он может работать — в коде нет привязки к POWER9, а
>   ветки для sm80+ (bf16, int8/fp8/nvfp4-ядра, Triton INT8) написаны, — но это не проверено.
> * **Поддержка image-моделей, LoRA для MiniMax H3, PDD-чекпойнтов H3 и Context Windows для H3 добавлена в этой
>   версии и на реальном железе ещё не запускалась** (написана без GPU, проверена тестами без CUDA). Перед работой
>   прогоните тесты из раздела 10.
> * Пакет распространяется под **GPL-3.0-or-later** без каких-либо гарантий (см. [LICENSE](LICENSE),
>   сторонние компоненты — [NOTICE](NOTICE)). Веса моделей не входят в пакет.

PowerShard считает одну большую видео- или image-модель **на нескольких GPU одновременно**: веса делятся между картами
(FSDP2), длинная последовательность видео-токенов — тоже (sequence parallel: token или Ulysses). Каждая
карта держит только свою часть, поэтому 14–22B модели и длинные видео помещаются в 6 × 16 GB.

| Что | Модели |
|---|---|
| **Wan** | 2.2 A14B T2V / I2V (MoE high+low), TI2V-5B, S2V, Animate, WanDancer; 2.1 T2V / I2V / FLF, VACE, HuMo, SCAIL, Animate2, InfiniteTalk / MultiTalk, Fun Camera / Control, Uni3C |
| **LTX** | LTX-2 / 2.3 / 2.5 видео+звук (LTXAV) и только видео (LTX-Video) |
| **MiniMax H3** | T2VA / FL2VA / R2VA; квантованные веса; **Veda Sparse Attention** |
| **Image** | любая модель, которую загружает ваша ComfyUI: SDXL / SD 1.5, Flux.1 / Kontext / Krea, Flux.2, Chroma, Z-Image, Lumina 2, Qwen-Image (Edit), MOSS-Image, MiniMax H3-Image, HiDream, SD3.5, Fibo 2 и др. Sequence parallel — Flux / Flux.2 / Chroma, Qwen-Image, Z-Image / Lumina 2, MOSS-Image |
| **Текстовые энкодеры** | umT5-XXL (Wan), Gemma (LTX), Qwen3-VL 32B (H3), T5 / CLIP (Flux), Qwen2.5-VL (Qwen-Image), Qwen3-4B (Z-Image), Qwen3.5-9B (MOSS), Mistral (Flux.2) — тоже по GPU |
| **Ускорители** | Block Cache (TeaCache/FBCache), NAG, RIFLEx; совместимы EasyCache/LazyCache, Context Windows, STG |

---

## 1. Установка

1. Скопируйте папку в `ComfyUI/custom_nodes/ComfyUI-PowerShard` (одна копия!) и перезапустите ComfyUI.
2. Новых зависимостей нет — используются ваши torch / NCCL / flash-attn / Triton. Нужна свежая ComfyUI с
   поддержкой нужной модели (Wan 2.2, LTX-2, MiniMax H3). Модели и энкодеры строит код вашей ComfyUI, поэтому
   новые семейства требуют её обновления: MOSS-Image — v0.42+, PDD-чекпойнты H3 — v0.42+, Flux.3 — v0.41+,
   Fibo 2 — v0.43+. Проверка окружения: `python scripts/diagnose.py`.
3. Модели — в стандартные папки ComfyUI:

| Файлы | Папка |
|---|---|
| Wan (`wan2.2_*_14B_*`, `wan2.2_ti2v_5B_*`, Wan 2.1), MiniMax H3, image-модели (Flux, Z-Image, Qwen-Image, MOSS...) | `models/diffusion_models` |
| SDXL / SD 1.5 и другие полные checkpoints | `models/checkpoints` |
| LTX: полный checkpoint (`ltx-2.3-22b-dev.safetensors`) или только трансформер | `models/checkpoints` или `models/diffusion_models` |
| umT5-XXL, Gemma 3 12B / Gemma 4, Qwen3-VL 32B (H3) | `models/text_encoders` |
| VAE Wan / H3 | `models/vae` |
| LoRA | `models/loras` |
| Uni3C, InfiniteTalk | `models/model_patches` |
| Предиктор Veda | `models/veda` |

Форматы файлов — любые, что понимает ComfyUI: bf16 / fp16 / fp32, fp8 (scaled и `comfy_quant`), int8, int4
(ConvRot / asym / w6a8), nvfp4, mxfp8. GGUF не поддерживается.

## 2. Готовые workflows

Папка `example_workflows/` (UI-формат — перетащите файл в окно ComfyUI; в новых версиях ComfyUI они также
видны в меню шаблонов). Имена моделей в них — примеры, выберите свои файлы.

| Файл | Что делает |
|---|---|
| `wan22_t2v_a14b.json` | Wan 2.2 A14B text-to-video, 832×480, 81 кадр, оба эксперта в одном пуле, один KSampler |
| `wan22_i2v_a14b.json` | то же из картинки (`WanImageToVideo`) |
| `wan22_ti2v_5b.json` | Wan 2.2 TI2V 5B, 1280×704, 121 кадр; без картинки — text-to-video |
| `ltx2_t2v_audio.json` | LTX-2.3 видео со звуком из текста |
| `ltx2_i2v_audio.json` | LTX-2.3 видео со звуком из картинки |
| `minimax_h3_t2va.json` | MiniMax H3 видео+звук, int8-веса без распаковки в VRAM |
| `minimax_h3_t2va_veda.json` | то же + Veda Sparse Attention |
| `sdxl_t2i.json` | SDXL 1024×1024: UNet в workers (batch parallel), CLIP и VAE из того же checkpoint |
| `flux1_dev_t2i.json` | Flux.1 dev: токены картинки по GPU, clip_l + t5xxl в workers |
| `z_image_turbo_t2i.json` | Z-Image Turbo, 8 шагов |
| `moss_image_t2i.json` | MOSS-Image, 50 шагов, cfg 3 (латенты Flux.2) |
| `qwen_image_t2i.json` | Qwen-Image 20B, 1328×1328 |

Все графы используют `sequence_mode=ulysses`, `gpu_ids=all` и `PowerShard: освободить GPU перед VAE`.
Пересобрать: `python scripts/build_example_workflows.py`.

## 3. Ноды

Минимальная цепочка: **`PowerShard: GPU / память / attention` → loader → обычные ноды ComfyUI**. Остальное —
по необходимости.

### Общие

| Нода | Назначение |
|---|---|
| `PowerShard: GPU / память / attention` | GPU (`all` или `5,2,0` — порядок задаёт ranks), где лежат веса (`gpu` — VRAM, `cpu` — pinned RAM, `ats` — эксперимент), attention backend, `token` / `ulysses` |
| `PowerShard: дополнительные настройки` | резерв VRAM, prefetch, NUMA, strict attention, `allow_host_wrappers`, fp16-обмен — обычно не нужна |
| `PowerShard: формат весов / вычисление` | `weight_format`, `compute`, `weight_dtype`, `fp16_accumulation` (раздел 5). Подключается ко входу `weights` любого loader-а; не подключена — fp16/bf16 как в обычной ComfyUI |
| `PowerShard: LoRA` | LoRA / LoHa / LoKr для любого loader-а (Wan, LTX, H3, image); цепочкой — несколько LoRA (раздел 6) |
| `PowerShard: Text Encoder` | `type`: `wan_umt5`, `ltx_gemma`, `flux`, `flux2`, `qwen_image`, `z_image`, `moss_image`, `sdxl`, `sd3`, `hidream` …; `placement`: `workers` (по GPU) или `cpu_fp32` (CPU, FP32) |
| `PowerShard: освободить GPU перед VAE` | паркует/закрывает workers перед декодированием |
| `PowerShard: Free VRAM` | выгружает с cuda:0 родные модели ComfyUI (CLIP vision, VAE) перед sampler |
| `PowerShard: checkpoint info (Wan / LTX / image)` | семейство, геометрия, формат, оценка памяти без загрузки весов |
| `PowerShard: диагностика среды` | GPU, torch, NCCL, attention-провайдеры |

У всех loader-ов есть **`keep_in_memory`**: `true` (по умолчанию) — между задачами веса остаются кусками в RAM
workers, VRAM освобождается, следующая задача стартует без чтения файла; `false` — workers закрываются
после каждой задачи.

### Wan

| Нода | Назначение |
|---|---|
| `PowerShard Wan 2.2: MoE Loader (high+low)` | два эксперта A14B в одном пуле. Выход `moe` — для одного sampler (эксперт переключается сам; `boundary=0` — авто: T2V 0.875, I2V 0.900); `high_noise` / `low_noise` — для двух `KSamplerAdvanced` |
| `PowerShard Wan: Sequence Loader (одна модель)` | TI2V-5B, Wan 2.1 и варианты, один эксперт |
| `PowerShard Wan: численная политика / память` | `fp16_safe`, `mlp_chunk_mode` / `mlp_chunk_tokens`, `moe_residency` (`swap` / `both`), `batch_chunk`, `model_type` |
| `PowerShard Wan: Uni3C ControlNet`, `… InfiniteTalk / MultiTalk`, `… Animate2 Cache` | специальные режимы |

Вариант модели определяется по заголовку файла; conditioning делают родные ноды (`WanImageToVideo`,
`Wan22ImageToVideoLatent`, `WanVaceToVideo`, `WanSoundImageToVideo`, `WanAnimateToVideo`, `WanHuMoImageToVideo`,
`WanSCAILToVideo`, `WanDancerVideo`, `WanAnimate2ToVideo`, `WanCameraImageToVideo`, `WanFunControlToVideo`).

`moe_residency=swap` (по умолчанию): в VRAM один эксперт, второй в RAM; `both` — оба в VRAM (~9.5 GB весов
на V100 при `gpu`, мало места под активации).

### LTX-2

| Нода | Назначение |
|---|---|
| `PowerShard LTX-2/2.5: Sequence Loader` | полный checkpoint или только трансформер |
| `PowerShard LTX: численная политика / память` | `fp16_safe`, chunk FFN, `batch_chunk`, `attention_chunk` |
| `PowerShard LTX: Video+Audio VAE из checkpoint` | VAE видео и звука + vocoder без чтения весов трансформера |

Видео-токены делятся по GPU, аудио-токены (сотни) — копия на каждом GPU; audio→video attention считается
точной частичной softmax + all-reduce. Работают с родными LTXV нодами: Context Windows, EasyCache / LazyCache,
STG, Modality Guidance, DualCFGGuider, guides / keyframes / IC-LoRA, латентный апскейлер. Для 6 × V100:
`weight_placement=gpu` (22B fp16 ≈ 7.5 GB на GPU) до ~121 кадра 768×512; длиннее — `cpu` + `mlp_chunk_mode=auto`.

### MiniMax H3

| Нода | Назначение |
|---|---|
| `PowerShard MiniMax H3: Loader` | H3 с любым форматом весов (вход `weights`), LoRA (вход `lora`: PTD, FastH3, PDD-LoRA и др.), PDD-чекпойнты, Context Windows / EasyCache, Cancel внутри шага, Veda |
| `PowerShard MiniMax H3: Veda Sparse Attention` | разреженная attention (ниже) |
| `PowerShard H3 Qwen Loader` | Qwen3-VL 32B text encoder H3 в пуле workers |
| `PowerShard: H3 Sequence Loader`, `… FP16 Patcher`, `… H3 MLP chunk`, `… Qwen MLP chunk`, `PowerShard Spectrum` | исходный путь PowerShard 0.5.3 (без изменений) |

Новое для H3 в этой версии (вслед за ComfyUI v0.41–0.43):

* **LoRA** — ключи разбирает родной `comfy.lora` (в т.ч. одноголовые LoRA для банков PDD-голов), дельта
  сливается в локальные строки каждого GPU. Не сливается в native INT8 (`precision=int8_fp16`): выберите `fp16`
  и формат `int8` в узле формата весов.
* **PDD-чекпойнты** (`pdd_steps` в metadata) — стопки выходных голов; нужна ComfyUI v0.42+.
* **Context Windows и EasyCache / LazyCache** — работают на host, как с Wan/LTX.
* Родные `MiniMax H3 Latent Upscale With Model`, LightVAE и Sol-H3-VAE работают на host без изменений.

#### Veda Sparse Attention

[Veda](https://github.com/veda-sparse/Veda-on-ComfyUI) (ICML 2026): маленький обученный предиктор оценивает
3D-тайлы по 128 токенов, и attention видео×видео считается только для лучших тайлов (по умолчанию 32 тайла
ключей на тайл запроса ≈ 10%). Текст и аудио остаются плотными, веса H3 не меняются.

1. Предиктор (~275 MB) положите в `models/veda/`:
   [T2VA](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) или
   [R2VA](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview).
2. В `PowerShard: GPU / память / attention` — **`sequence_mode=ulysses`** (обязательно при нескольких GPU).
3. `PowerShard MiniMax H3: Loader` → `PowerShard MiniMax H3: Veda Sparse Attention` → sampler.

| Вход | По умолчанию | Смысл |
|---|---|---|
| `generated_sparsity` | `32` | тайлов ключей на тайл запроса; `90%` — доля пропуска; `trained` — как при обучении |
| `reference_sparsity` | `32` | то же для референсов (первый/последний кадр, reference); `0%` — полная attention |
| `full_attention_layers` / `full_attention_steps` | пусто | блоки / шаги с полной attention, например `0, 1, 47-49` |
| `backend` | `auto` | `auto`: на V100 — `torch-sdpa`, на sm80+ — самый быстрый из прошедших проверку. `triton-int8` — sm80+; `triton-fp16` — Triton без tensor cores на V100 (медленно, для сравнения); `torch-sdpa` — без Triton |

Смена настроек Veda не перезапускает workers. Первый шаг включает загрузку предиктора и компиляцию ядер.
На V100 Triton 3.x не использует tensor cores Volta, поэтому `triton-fp16` там медленный (GPU 100%, ~150 Вт
из 300) — используйте `auto` / `torch-sdpa`.

### Image-модели

| Нода | Назначение |
|---|---|
| `PowerShard Image: Loader` | любая image-модель ComfyUI (`diffusion_models/` или полный checkpoint из `checkpoints/`); входы `parallel`, `weights`, `lora`, `keep_in_memory` |
| `PowerShard Image: CLIP + VAE из checkpoint` | CLIP и VAE полного checkpoint (SDXL, SD 1.5) на host — без загрузки UNet |

Модель строит сама ComfyUI (определение семейства, `model_sampling`, conditioning — родные), forward — родной
код модели; PowerShard шардирует веса по GPU (FSDP2) и делит вычисление:

| `parallel` | Как | Для чего |
|---|---|---|
| `sequence` | токены картинки по GPU, ключи/значения attention собираются со всех GPU | Flux / Flux.2 / Chroma, Qwen-Image, Z-Image / Lumina 2, MOSS-Image — ускорение и при batch 1 |
| `batch` | строки batch (cond / uncond, batch_size) по GPU | любая модель, в т.ч. SDXL; с CFG делит работу на 2 GPU, с batch_size 3 — на 6 |
| `replicate` | все GPU считают одно и то же | только экономия памяти (модель не помещается в одну карту) |
| `auto` | `sequence`, если семейство поддерживает, иначе `batch` | по умолчанию |

dtype (`weight_dtype` в узле формата весов): `auto` — bf16 на sm80+, fp16 на V100, если модель его поддерживает
(Flux, Chroma, SDXL); модели только с bf16 (Z-Image, Qwen-Image, MOSS-Image, HiDream) на V100 — bf16 хранение +
точное FP32 вычисление (медленнее; `weight_dtype=fp16` быстрее, но качество нужно проверить). Квантованные файлы
(fp8 / int8 / nvfp4 / int4) и квантование при загрузке — как у Wan/LTX.

Работают: родные ControlNet (считаются на host, результат уходит в workers), EasyCache / LazyCache, CFG-ноды
(CFGNorm, APG и т.п.), сдвиги `ModelSampling*`, Flux Kontext / Qwen-Image Edit / MOSS-Image Edit (референсы),
Skip Layer Guidance (`SkipLayerGuidanceDiT` / `SkipLayerGuidanceDiTSimple`): список пропускаемых блоков уходит в
workers. SLG нужна модель, чей родной forward поддерживает замену блоков (Flux, Chroma, Qwen-Image, MOSS-Image);
у Lumina 2 / Z-Image её нет.
Не работают: IP-Adapter, PuLID и другие патчи attention/блоков (`patches`, прочие `patches_replace`), родной
`LoraLoader` (используйте `PowerShard: LoRA`), `CLIPSetLastLayer` с distributed-энкодером (для SD 1.5 с clip skip
берите родной `CLIPLoader`).

### Ускорители

| Нода | Что делает |
|---|---|
| `PowerShard: Block Cache` | TeaCache/FBCache: блоки пропускаются, если residual блока 0 изменился меньше `threshold` (0.05–0.12). Решение одно на все GPU. `max_cache_mb` — предел VRAM кэша на GPU |
| `PowerShard: NAG` | негативный промпт без CFG (distilled / Lightning при cfg=1) |
| `PowerShard Wan: RIFLEx` | видео длиннее 81 кадра без повтора движения |

Сторонние патчи `forward` (TeaCache/MagCache сообщества, SageAttention-патч KJ) в workers не переносятся —
будет понятная ошибка.

## 4. Текстовые энкодеры

`PowerShard: Text Encoder` с `placement=workers` держит энкодер на тех же GPU, что генератор, и считает в FP32 —
cuda:0 не занимается 11–24 GB энкодера. `after_encode`: `release` — workers закрываются при старте генератора;
`release_now` — сразу после encode; `keep_ram` — workers живут между генерациями: при старте генератора веса паркуются в RAM и возвращаются на GPU без чтения файла. Одинаковый
промпт берётся из кэша. Для `ltx_gemma` выберите `ltx_checkpoint` (файл LTX с text projection). Для энкодеров
из нескольких файлов (`flux` — clip_l + t5xxl, `sdxl` — clip_l + clip_g, `sd3`, `hidream`) остальные файлы —
во входах `text_encoder_2…4`. Маленькие CLIP (SDXL) проще оставить родному `CLIPLoader`.
`placement=cpu_fp32` — родной энкодер ComfyUI на CPU в FP32 (медленнее, но не трогает GPU).

## 5. Формат весов и вычисление

Узел `PowerShard: формат весов / вычисление`:

| Параметр | Значения |
|---|---|
| `weight_format` | `dequantize` — fp16/bf16 (по умолчанию); `as_file` — как в файле (int8/int4/nvfp4/fp8 остаются квантованными); `int8` / `fp8` / `mxfp8` / `nvfp4` / `int4` — квантовать при загрузке |
| `compute` | `auto` — native-ядра, если GPU и comfy-kitchen умеют формат, иначе деквантование; `native`; `dequantize` — вес на время GEMM переводится в fp16/bf16 (работает везде) |
| `weight_dtype` | `auto` — bf16 на sm80+, fp16 на V100/T4 (image-модели без fp16 на V100 — bf16 + FP32 вычисление); `fp16`; `bf16`; `fp32` (только image). H3 всегда fp16 |
| `fp16_accumulation` | cuBLAS накапливает fp16 GEMM в fp16: до 2× только на RTX 30/40/50 с fp16-весами; на V100/A100/H100 эффекта нет, точность ниже |

Native-ядра: int8/int4 — sm75+ и CUDA-backend comfy-kitchen (torch cu130+); fp8 — sm89+; nvfp4/mxfp8 — sm100+.
На AC922 (CUDA 12.4) всегда деквантование. Рекомендации: 6 × V100 — `dequantize` (быстрее) или `as_file` /
`int8` + `compute=dequantize`, если не хватает VRAM; RTX 5090 — `as_file` или `nvfp4` / `fp8`, `compute=auto`.

## 6. LoRA

Один узел `PowerShard: LoRA` для всех loader-ов (тип `POWERSHARD_LORA`, цепочка — несколько LoRA; `apply_to` —
только для экспертов Wan 2.2 MoE). Старые `PowerShard Wan/LTX: LoRA` скрыты из поиска, но открываются в старых
workflows.

LoRA вливается при загрузке прямо в локальные строки каждого GPU (`W[a:b] += s · up[a:b] @ down`), полный вес
не собирается; смена LoRA = перезагрузка workers.

* Wan / LTX: kohya / diffusers / PEFT / mochi, LoHa, LoKr, `.diff` / `.diff_b`; ключи, не подходящие модели,
  пропускаются с предупреждением.
* MiniMax H3 и image-модели: ключи сопоставляет родной `comfy.lora` (все форматы, которые понимает ComfyUI для
  этой модели, включая fused qkv Flux и PDD-банки H3), дельта считается его адаптерами.
* DoRA, OFT, BOFT, GLoRA, LoCon-mid и Tucker зависят от полного веса и отклоняются явно. Родной
  `LoraLoaderModelOnly` на PowerShard-модели выдаёт ошибку с подсказкой.

## 7. Отмена задачи (Cancel) и VRAM

* **Cancel во время шага:** workers Wan, LTX, image-моделей и `PowerShard MiniMax H3: Loader` проверяют отмену перед каждым
  DiT-блоком и выходят на одном блоке на всех GPU — без ожидания конца шага. Исходный H3 Sequence Loader
  дожидается конца шага.
* **Повторный Cancel** во время дренажа — принудительный сброс: workers останавливаются, VRAM освобождается.
* **«Unload models» / «Free model and node cache»** в ComfyUI и `--disable-smart-memory` останавливают все
  PowerShard workers.
* Зависший rank: дренаж ограничен по времени, NCCL-операции — таймаутом (переменные ниже).

`nvidia-smi` может показывать занятую память, не привязанную ни к одному процессу: так драйвер отображает
буферы NCCL (cuMem) и `expandable_segments` тех же workers. Если процессов-сирот нет (`ps -ef | grep
wan_worker`), память освободится вместе с workers.

## 8. Память и скорость

* Первая задача: запуск workers, проверка attention, чтение checkpoint. Повторные с `keep_in_memory=true`
  файл не читают. Смена config / LoRA / опций — новая загрузка.
* `mlp_chunk_mode=auto` режет FFN по токенам по свободной VRAM — рекомендован для 720p на 16 GB.
* Ulysses рекомендован для Wan 14B (40 голов → 7 на GPU при 6 GPU) и длинных видео; `token` собирает полный
  K/V на каждом GPU.
* `weight_placement=ats` — эксперимент (POWER9 + NVLink ATS). Для V100 обычно `gpu` или `cpu`.
* Родные модели ComfyUI (VAE, CLIP vision) живут на cuda:0, где и rank 0 — используйте энкодеры в workers и
  `Free VRAM`.

## 9. Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `POWERSHARD_DRAIN_TIMEOUT_S` | авто (≥ 1200) | сколько ждать окончания шага после Cancel |
| `POWERSHARD_NCCL_TIMEOUT_MIN` | 60 | таймаут одной NCCL-операции в workers |
| `POWERSHARD_SHUTDOWN_GRACE_S` | 3 | ожидание штатного выхода workers перед SIGTERM |
| `POWERSHARD_KEEP_REPORTS` | 300 | сколько файлов отчётов хранить в `output/powershard/` |
| `POWERSHARD_CANCEL_CHECK_BLOCKS` | 1 | проверять Cancel каждые N блоков |
| `POWERSHARD_DEBUG_FINITE` | 0 | `1` — проверка NaN/Inf по блокам (диагностика, медленно) |
| `POWERSHARD_PROFILE=1` | выкл. | torch.profiler трасса первых forward |

Логи ranks: `<ComfyUI>/output/powershard/` (`*-rankN.log`, `*-progress.json`, `run-*.json`).

## 10. Проверка на своём сервере

Из папки custom node, Python вашей ComfyUI:

```bash
# без GPU: ноды, workflows, форматы, LoRA, жизненный цикл workers, настройки Veda
python -m pytest tests/test_wan_config.py tests/test_ltx_config.py tests/test_lifecycle_wan.py \
                 tests/test_veda_config.py tests/test_image_config.py tests/test_example_workflows.py
# image: sequence / batch parallel против обычного forward (CPU, gloo, родные Flux и Qwen-Image)
python -m pytest tests/test_image_parallel_torch.py --comfy /path/to/ComfyUI
# CPU + torch: эквивалентность с родными моделями ComfyUI
python -m pytest tests/test_wan_native.py tests/test_wan_pipeline.py tests/test_ltx_native.py \
                 tests/test_quant_formats_torch.py --comfy /path/to/ComfyUI
python -m pytest tests/test_lora_torch.py tests/test_veda_torch.py
# CUDA/NCCL сквозная проверка
python scripts/probe_wan_cuda.py --comfy /path/to/ComfyUI --gpus all --sequence-mode ulysses --moe
python scripts/probe_ltx_cuda.py --comfy /path/to/ComfyUI --gpus all --sequence-mode ulysses
python scripts/probe_veda_cuda.py --gpu 0 --grid 37x24x42 --heads 7 --predictor /path/to/models/veda/<file>.safetensors
```

## 11. Ограничения

* Проверено только на AC922 / 6 × V100 (см. начало). Остальные GPU и x86_64 — не проверены.
* Не поддерживаются: GGUF, torch.compile, сторонние `patches_replace`, Spectrum для Wan/LTX, in-context latents.
* Veda: только MiniMax H3 и только через `PowerShard MiniMax H3: Loader`; при нескольких GPU — Ulysses.
* Image: sequence parallel только для перечисленных семейств; в режиме `sequence` attention считается
  глобальным бэкендом ComfyUI (per-layer `comfy_attention` из файла модели не используется). Qwen-Image 2.1
  работает без prefix-кэша (batch / replicate). IP-Adapter / PuLID / патчи блоков не переносятся в workers.
* Block Cache — эвристика: при артефактах уменьшите `threshold` или `max_consecutive_skips`.
* Animate2: без Block Cache / NAG / RIFLEx. InfiniteTalk и Uni3C — только Wan T2V/I2V.
* Оценки памяти — оценки, не гарантия.

## 12. Лицензия и происхождение

* Код пакета: **GPL-3.0-or-later** ([LICENSE](LICENSE)). Это изменённая версия ComfyUI-PowerShard 0.5.3: файлы
  0.5.3 не изменены, всё новое добавлено отдельными файлами и строками в конце `__init__.py`; полный diff —
  `review/changes_wan22.patch`.
* Сторонний код и идеи — [NOTICE](NOTICE) и `licenses/`: ComfyUI (GPL-3.0), Veda-on-ComfyUI и Miowtion (MIT),
  SageAttention (BSD-3 / Apache-2.0), Comfy Kitchen и Raylight (Apache-2.0), Spectrum (GPL-3.0), MIT-проекты
  FP16-исправлений H3.
* Веса моделей не входят в пакет и используются по своим лицензиям (Wan — Apache-2.0, LTX-2 — LTX-2 Community
  License, MiniMax H3 и предикторы Veda — MiniMax H3 Community License, Gemma — Gemma Terms of Use, Qwen3-VL и
  umT5 — Apache-2.0; image: SDXL — CreativeML Open RAIL++-M, Flux.1 dev / Flux.2 dev — FLUX Non-Commercial, Z-Image /
  Qwen-Image / MOSS-Image — Apache-2.0). Принимать их условия — ответственность пользователя.
* Названия ComfyUI, Wan, LTX, MiniMax, NVIDIA, IBM — товарные знаки владельцев; проект с ними не связан.

## 13. Файлы

| Путь | Что внутри |
|---|---|
| `powershard/wan_*.py`, `ltx_*.py` | Wan и LTX: заголовки, модель, backend, worker, адаптер ComfyUI, ноды |
| `powershard/common_nodes.py` | формат весов, LoRA, text encoder, checkpoint info |
| `powershard/image_*.py`, `comfy_lora.py` | image-модели: host-proxy, worker, sequence/batch parallel, LoRA через comfy.lora |
| `powershard/native_te.py`, `wan_text.py` | энкодеры в workers |
| `powershard/h3_quant.py`, `veda/`, `veda_nodes.py` | MiniMax H3 loader и Veda |
| `powershard/quant_*.py` | форматы весов, квантованные Linear в FSDP |
| `powershard/accel.py` | Block Cache, NAG, RIFLEx |
| `powershard/lifecycle.py`, `cancel.py` | отмена задачи, дренаж, остановка workers |
| остальное в `powershard/` | исходный PowerShard 0.5.3 (H3, Qwen, config, attention, FSDP) |
| `example_workflows/` | готовые графы; `scripts/build_example_workflows.py` — их генератор |
| `scripts/`, `tests/` | пробы CUDA, диагностика, тесты |
| `docs/history/` | исторические отчёты 0.2–0.5.3 |
