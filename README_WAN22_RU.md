# PowerShard Wan 2.1 / 2.2 — дополнение к ComfyUI-PowerShard 0.5.3

Wan 2.2 (A14B T2V/I2V MoE, TI2V-5B, **S2V**, **Animate**, **WanDancer**), Wan 2.1 T2V/I2V/FLF, **VACE**, **HuMo**, **SCAIL / SCAIL-2**, **Animate2**, **InfiniteTalk / MultiTalk**, **Fun Camera / Fun Control** и **Uni3C ControlNet** на выбранном наборе GPU тем же механизмом, что H3: отдельный subprocess на каждый CUDA UUID, FSDP2 Shard(0) весов (VRAM / pinned RAM / ATS), sequence parallel по токенам видео (token или Ulysses), RAM-парковка между задачами. Обоснование выбора и бюджет памяти: [RESEARCH_WAN22_RU.md](RESEARCH_WAN22_RU.md).

**Существующий код не изменён.** Добавлены только новые файлы и 8 строк в конце `__init__.py` (регистрация нод). H3/Qwen ноды, workflows и тесты работают как раньше.

## Установка

Как обычное обновление PowerShard: заменить папку `ComfyUI/custom_nodes/ComfyUI-PowerShard`, перезапустить ComfyUI. Новых зависимостей нет (torch/NCCL/FA2 ваши). Модели — в стандартные папки ComfyUI:

| Файл (Comfy-Org/Wan_2.2_ComfyUI_Repackaged) | Папка |
|---|---|
| `wan2.2_t2v_high_noise_14B_fp16.safetensors`, `..._low_noise_...` (или `fp8_scaled`) | `models/diffusion_models` |
| `wan2.2_i2v_high_noise_14B_*.safetensors`, `..._low_noise_...` | `models/diffusion_models` |
| `wan2.2_ti2v_5B_fp16.safetensors` | `models/diffusion_models` |
| `umt5_xxl_fp16.safetensors` | `models/text_encoders` |
| `wan_2.1_vae.safetensors` (A14B), `wan2.2_vae.safetensors` (5B) | `models/vae` |
| `wan2.2_*_lightx2v_4steps_lora_*_{high,low}_noise.safetensors` (опционально) | `models/loras` |

`fp8_scaled`/`fp8` чекпоинты читаются и деквантуются в FP16 по локальным строкам шардов (V100 не считает в fp8; экономится диск и чтение, не VRAM). В `PowerShardConfig` для них `precision=fp16`; `int8_fp16` — только для native int8_tensorwise чекпоинтов.

## Ноды (категория PowerShard/Wan)

| Нода | Назначение |
|---|---|
| **PowerShard Wan 2.2: MoE Loader (high+low)** | два эксперта A14B в одном пуле workers. Выходы `high_noise`, `low_noise` (для двух `KSamplerAdvanced`) и `moe` (один sampler, переключение по `boundary`: T2V 0.875, I2V 0.900) |
| **PowerShard Wan: Sequence Loader (одна модель)** | TI2V-5B, Wan 2.1, либо один эксперт |
| **PowerShard Wan: численная политика / память** (`WAN_OPTIONS`) | `fp16_safe` (scaled GEMM, при NaN), `mlp_chunk_mode/tokens`, `moe_residency` (`swap`/`both`), `batch_chunk`, `model_type` (`auto` / `animate2`) |
| **PowerShard Wan: LoRA (слияние в shards)** (`WAN_LORA`) | цепочка LoRA; `apply_to`: all / high_noise / low_noise |
| **PowerShard Wan: umT5 Distributed Encoder** | umT5-XXL в пуле PowerShard workers: веса FSDP по GPU из того же `PowerShardConfig` (VRAM или pinned RAM), вычисление FP32, родной comfy encode (маски, веса токенов). `after_encode=release` (по умолчанию): все промпты запуска (positive, negative, pose) кодируются одним стартом workers, которые закрываются при старте генератора PowerShard или нодой Free VRAM — cuda:0 не занимается 11 GB umT5; `release_now` — закрывать после каждого encode; `keep_ram` — shards в pinned RAM. Результат кэшируется на host по тексту и файлу энкодера (общий кэш, переживает пересоздание ноды и смену config): одинаковый текст не кодируется повторно. Имя файла не проверяется — проверяется структура umT5-XXL. **Рекомендуется вместо native/`native_offload`** |
| **PowerShard Wan: umT5 Text Encoder** | umT5-XXL в FP32 на CPU (`cpu_fp32`) или native (`native_offload` кладёт 11 GB на cuda:0 — это и rank 0 workers) |
| **PowerShard: Free VRAM** | пропускает `positive`/`negative`/`latent` насквозь и перед этим выгружает с GPU подключённые `clip` / `clip_vision` / `vae` (или все native модели: `mode=all_native`; PowerShard-модели не трогаются). Ставится между `…ToVideo` и sampler. Distributed umT5 при этом закрывает свои workers |
| **PowerShard Wan: Animate2 Cache (RAM workers)** | кэш входов pose branch Animate2 в RAM workers (fp16/fp32), без host-объектов и callbacks. Native `WanAnimate2Cache` тоже принимается (его host-кэш не используется, берутся только dtype/идентичность) |
| **PowerShard Wan: checkpoint info** | геометрия, вариант (vace/s2v/animate/camera), формат (fp16/bf16/fp8_scaled), оценка памяти на GPU — без загрузки весов |
| **PowerShard Wan: Uni3C ControlNet** | ControlNet для Wan в ComfyUI (Uni3C, файл из `models/model_patches`): render video → render latent на host, веса ControlNet шардируются в workers. Замена native «Apply Wan Uni3C ControlNet» |
| **PowerShard Wan: InfiniteTalk / MultiTalk** | замена native `WanInfiniteTalkToVideo` с теми же входами (1 или 2 говорящих, маски, `previous_frames` для продолжения), но вместо `MODEL_PATCH` — имя файла из `models/model_patches`. Логика кадров/аудио — родной код comfy; 40 аудио-блоков шардируются в workers, `audio_proj` считается на host |

### Варианты моделей

Вариант определяется по заголовку checkpoint (как `comfy.model_detection`), поэтому отдельных loaders нет: все варианты грузятся `Sequence Loader`, а conditioning делают родные ноды ComfyUI. Исключение — **Animate2**: его checkpoint имеет форму Wan 2.1 I2V, поэтому тип берётся из metadata (`config.transformer.model_type`) или задаётся `model_type=animate2` в `PowerShard Wan: численная политика`.

| Вариант | Родные ноды conditioning | Как распределено по GPU |
|---|---|---|
| **VACE** (Wan 2.1 VACE 1.3B/14B, в т.ч. i2v) | `WanVaceToVideo` → sampler → `TrimVideoLatent` | VACE-токены режутся по тем же строкам, что видео; VACE-блоки — отдельные FSDP units с той же распределённой self-attention; несколько контекстов и `vace_strength` |
| **S2V** (Wan 2.2 Sound-to-Video 14B) | `AudioEncoderLoader`/`AudioEncoderEncode` (wav2vec2) → `WanSoundImageToVideo` | аудио-энкодер и FramePack motion реплицированы (малые); 12 аудио-инъекторов считаются покадрово по своим строкам; ref/motion токены в хвосте последовательности |
| **Animate** (Wan 2.2 Animate 14B) | `CLIPVisionEncode` + `WanAnimateToVideo` → sampler → `TrimVideoLatent` | motion encoder лица (512×512) и pose embedding реплицированы; face adapter (каждый 5-й блок) покадрово по своим строкам |
| **Fun Camera** (2.1 / 2.2) | `WanCameraImageToVideo` | camera adapter добавляется к patch embedding до разбиения |
| **Fun Control / Fun Inpaint** | `WanFunControlToVideo`, `Wan22FunControlToVideo` | работает как обычная модель: control входит через concat-каналы |
| **Uni3C ControlNet** | `PowerShard Wan: Uni3C ControlNet` | отдельный FSDP root в тех же процессах, строки токенов совпадают с основной моделью, residual добавляется после блоков 0…N-1 |
| **HuMo** (17B и 1.7B) | `AudioEncoderEncode` (whisper) + `WanHuMoImageToVideo` | `audio_proj` — FSDP unit (реплицированный вызов); аудио cross-attention внутри каждого блока покадрово (16 аудио-токенов на кадр) по своим строкам; ref-кадры в хвосте последовательности, аудио для них дополняется нулями как в native |
| **SCAIL / SCAIL-2** | `WanSCAILToVideo` (pose render, маски, replacement mode) | референс-кадры склеиваются по времени до patch embedding, pose-токены (в т.ч. половинного разрешения) — в хвосте; RoPE строит родной `SCAILWanModel.rope_encode` (animation / replacement); SCAIL-2 mask stream — отдельный FSDP unit |
| **WanDancer** | `WanDancerEncodeAudio` + `WanDancerVideo` | fps ≠ 30 → `patch_embedding_global`/`head_global` и fps-RoPE; музыкальный энкодер (2 слоя, FP32) и интерполяция реплицированы; 8 music-инъекторов покадрово по своим строкам; `in_proj` из checkpoint читается смещением строк без копии |
| **Animate2** (Wan-Animate-2) | `WanAnimate2ToVideo` (+ `WanAnimate2Cache`) → sampler → `TrimVideoLatent` | pose branch и генерация шардируются **обе** и считаются одним вызовом FSDP unit на блок; кадр j генерации видит всю генерацию + кадр j-1 pose: token-режим собирает K/V обеих ветвей, Ulysses — all-to-all обеих; `pose_strength`, `reference_image_strength`, окно pose по шагам. `WanAnimate2Cache` → кэш входов pose branch в RAM workers (локальные строки, fp16), со 2-го шага pose branch не считается |
| **InfiniteTalk / MultiTalk** (на Wan 2.1 I2V 14B) | `AudioEncoderEncode` (wav2vec2) + `PowerShard Wan: InfiniteTalk / MultiTalk` | model patch — третий FSDP root, вызывается внутри блока после cross-attention (как native `attn2_patch`); 2 говорящих: карта внимания на маски считается в self-attention того же блока по shard-строкам (token: all-gather карты, Ulysses: all-reduce по heads) и даёт RoPE-позиции запросов |

Классического «ControlNet» (cond-level, как у SD/Flux) для Wan в ComfyUI нет: управление делается через VACE, Fun Control (concat) или Uni3C — все три поддержаны.

Таким образом поддержаны все варианты Wan из `comfy.model_detection`/`supported_models` текущей ComfyUI (t2v, i2v, flf, ti2v, vace, camera, camera_2.2, s2v, humo, animate, scail, scail2, wandancer, animate2) и оба model patch для Wan (Uni3C, InfiniteTalk/MultiTalk).

Используются обычные `PowerShardConfig` / `PowerShardConfigTuning` (GPU, placement, attention, token/Ulysses, wire dtype, prefetch) и `PowerShardRelease` перед VAE. Всё остальное — родные ноды ComfyUI: `CLIPTextEncode`, `ModelSamplingSD3`, `EmptyHunyuanLatentVideo`, `WanImageToVideo`, `Wan22ImageToVideoLatent`, `KSampler`/`KSamplerAdvanced`, `VAEDecodeTiled`, `CreateVideo`, `SaveVideo`.

## Готовые workflows (`workflows_wan/`)

| Файл | Что |
|---|---|
| `wan22_t2v_a14b_480p_gpu_ulysses` | T2V A14B 832×480×81, два KSamplerAdvanced (20 шагов, переход на 10), shift 8, CFG 3.5, placement=gpu, swap |
| `wan22_t2v_a14b_720p_cpu_ulysses` | 1280×720×81, веса в pinned RAM, `batch_chunk=1`, MLP auto, prefetch 1 |
| `wan22_t2v_a14b_lightning_4step` | Lightning LoRA high/low, 4 шага (2+2), CFG 1, shift 5 |
| `wan22_i2v_a14b_480p_gpu_ulysses` | I2V через `WanImageToVideo` (положите `example.png` в input) |
| `wan22_t2v_a14b_moe_single_sampler` | выход `moe` + один KSampler |
| `wan22_ti2v_5b_720p_i2v` / `_t2v` | 5B, 1280×704×121, 24 fps |
| `wan21_vace_14b_control.api.json` | VACE 14B, control video (depth/pose) через `WanVaceToVideo`, placement=cpu |
| `wan22_s2v_14b.api.json` | S2V: изображение + речь (`speech.wav`) → видео со звуком |
| `wan22_animate_14b.api.json` | Animate: персонаж + pose/face видео |
| `wan21_i2v_14b_uni3c_controlnet.api.json` | Wan 2.1 I2V 14B + Uni3C (render облака точек) |
| `wan21_humo_17b.api.json` | HuMo: фото человека + речь (whisper) → говорящее видео 97 кадров, 25 fps |
| `wan21_scail_14b.api.json` | SCAIL: персонаж + pose render видео, 512×896 |
| `wan22_wandancer_14b.api.json` | WanDancer: фото + музыка → танец, 149 кадров |
| `wan_animate2_14b.api.json` | Animate2 (`model_type=animate2`) + `WanAnimate2Cache`, `TrimVideoLatent` |
| `wan21_infinitetalk_14b.api.json` / `_two_speakers` | InfiniteTalk: 1 или 2 говорящих (маски `mask_speaker_1/2.png`) на Wan 2.1 I2V 14B |

Варианты сохранены только в API-формате (у `LoadVideo`/`LoadAudio` upload-виджеты зависят от версии frontend) — откройте перетаскиванием JSON в окно ComfyUI.

Пересборка: `python scripts/build_wan_workflows.py`. Имена файлов моделей — выберите свои в нодах.

## Время загрузки и первый шаг

Первый шаг после старта включает запуск workers: CUDA probes attention (раньше — последовательно 6–12 процессов, теперь параллельно и один раз на процесс ComfyUI), импорт torch/comfy в 6 процессах и загрузку весов (в логе rank: `load_s`, разбивка `load_timing`: `read_s` чтение файла, `convert_s` bf16/fp8→fp16, `lora_s` слияние LoRA, `store_s` запись в shards). Конвертация, LoRA и проверки весов теперь выполняются на GPU своего rank, а не на CPU POWER9.

* Первая загрузка после старта ComfyUI ограничена чтением checkpoint с диска (в логах: холодный файл ~50–90 с, из page cache ~25 с на rank; почти всё — `convert_s`, куда входит подкачка страниц файла). Повторные задачи с `keep_in_memory=true` файл не читают. Смена `PowerShardConfig` (cpu↔ats, GPU, attention), LoRA или опций — это новая session и новая загрузка.
* `mlp_chunk_mode=off` теперь означает «целиком, если помещается»: если оценка FFN не помещается (длинные видео), блок автоматически режется по токенам (в отчёте `override`). Раньше на 400+ кадрах это давало CUDA OOM.
* `keep_in_memory=true` в loader: после задачи веса остаются локальными shards в RAM (pinned при placement=cpu), следующая задача стартует без чтения файла и без конвертации. При `false` каждая задача заново читает checkpoint (у Animate2 bf16 это было ~145 с на rank).
* Чекпоинт в fp16 читается быстрее bf16/fp8 (нет преобразования); int8 — только с `precision=int8_fp16`.
* Host-модели ComfyUI (umT5, CLIP vision, VAE) по умолчанию сидят на cuda:0 — там же rank 0. Используйте `umT5 Distributed Encoder` и `Free VRAM`, иначе rank 0 работает почти без свободной памяти (в присланных логах — десятки MiB).

## Как это устроено

* **Один пул на оба эксперта.** Оба FSDP root живут в одних и тех же 6 процессах и NCCL group. `high_noise`/`low_noise`/`moe` — clones одного ModelPatcher, эксперт выбирается ключом в `transformer_options`, поэтому ComfyUI не выгружает модель при переходе между samplers.
* **Переход high → low.** После sampler-прохода, где работал только high-эксперт MoE, workers не паркуются и не закрываются (даже при `keep_in_memory=false`): очищается только conditioning. Low-проход продолжает с того же пула. После low-прохода — обычная политика (idle в RAM при keep, иначе закрытие).
* **`moe_residency`.** `swap` (по умолчанию): в VRAM активен один эксперт, неактивный при переключении паркуется в RAM (как keep между задачами; при placement=cpu это лишь reshard). `both`: оба эксперта активны — ~9.5 GB весов на V100 при placement=gpu, мало места под активации.
* **Sequence parallel.** Ulysses рекомендован: 40 heads → по 7 на ранк (паддинг до 42 на 6 GPU), обмен q/k/v all-to-all; token-режим собирает полный K/V на каждом ранке (дороже по памяти на 720p). Cross-attention к тексту локальна. После последнего блока head применяется к локальным токенам и собирается только его выход.
* **Численно.** FP32: residual, модуляция AdaLN, LayerNorm/RMSNorm, RoPE, time embedding/projection; FP16: GEMM и attention. q/k делятся на степени двойки из границ весов RMSNorm, V — по max, так что FP16 вход attention не переполняется. Ошибка non-finite на выходе поднимается явно с подсказкой включить `fp16_safe`.
* **Attention.** Тот же probe/policy механизм, что для H3, но с геометрией Wan: custom FA2 должен пройти проверку для всех heads (cross-attention) и `ceil(H/world)` (Ulysses), иначе используется SDPA/math. Фактические вызовы по группам `wan_self`/`wan_cross` — в статусе.
* **LoRA.** `W[a:b] += s·(alpha/r)·up[a:b] @ down` для локальных строк каждого ранка при загрузке (форматы `diffusion_model.*`, `lora_unet_*`, lora_up/down, lora_A/B, `.diff`, `.diff_b`). Смена LoRA = перезагрузка session. LoHa/LoKr/DoRA отклоняются явно. Native `LoraLoaderModelOnly` на PowerShard MODEL выдаёт ошибку с подсказкой.

## Проверка на сервере

Из папки custom node, Python вашей ComfyUI:

```bash
# Быстрые тесты без GPU (заголовки, LoRA-планирование, workflows)
python -m pytest tests/test_wan_config.py --comfy /opt/ComfyUI
# Численная эквивалентность с native comfy WanModel и MoE жизненный цикл (CPU)
python -m pytest tests/test_wan_native.py tests/test_wan_pipeline.py --comfy /opt/ComfyUI
# Настоящие gloo collectives: token/Ulysses на 2/3/4 ранках, FSDP2 WanBackend, все варианты на 3 ранках, fp16 wire на 4
python -m pytest tests/test_wan_distributed.py --comfy /opt/ComfyUI
# Сквозной CUDA/NCCL путь (малая случайная Wan, сравнение с FP32 native на CPU)
python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus all --sequence-mode ulysses --moe --fp8 --lora --output reports/wan-6gpu-gpu.json
python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus all --weight-placement cpu --moe --residency both --ram-roundtrip --output reports/wan-6gpu-cpu.json
python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2 --attention-backend sdpa --comm fp16 --output reports/wan-3gpu-fp16wire.json
# Варианты на настоящих GPU (малые модели против native FP32)
for v in vace s2v animate camera uni3c humo scail scail2 wandancer wandancer30 animate2 multitalk1 multitalk2; do
  python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus all --variant $v --output reports/wan-6gpu-$v.json
done
python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus all --variant animate2 --animate2-cache --output reports/wan-6gpu-animate2-cache.json
```

## Ограничения

* Не проверено на CUDA/NCCL/POWER9 в этой ревизии: в среде разработки не было torch (PyPI/HF закрыты политикой). Проверены статически, симуляцией раскладки Ulysses/token на numpy и torch-free тестами; GPU/gloo/native тесты приложены для запуска на сервере.
* Не поддерживаются: in-context latents (Bernini), сторонние attention/double_block patches, TeaCache, torch.compile, Spectrum для Wan. Native «Apply Wan Uni3C ControlNet» и native `WanInfiniteTalkToVideo` с `MODEL_PATCH` отклоняются с подсказкой использовать ноду PowerShard (веса patch должны грузиться в workers из файла).
* InfiniteTalk: native код работает только с batch 1 (CFG батчем падает); PowerShard обобщает — аудио-контекст общий для cond/uncond, карта говорящих считается для каждого элемента batch. При batch 1 совпадает с native. Uni3C и InfiniteTalk — только с Wan 2.1/2.2 T2V/I2V (не с HuMo/SCAIL/Animate2…).
* Animate2: `WanAnimate2Cache` с `dtype=int8/int4` в workers хранит fp16 (квантованный кэш comfy не переносится); context-window prepass не выполняется (он нужен native лишь для калибровки dynamic VRAM). Кэш живёт в RAM workers до смены cache-объекта/pose (до 2 слотов).
* WanDancer: `reference_latent` (ref_conv) отклоняется — оригинальный pipeline его не использует, а native RoPE с ним не согласован.
* Uni3C вместе с `reference_latent` (ref_conv) не поддерживается: сдвиг токенов. Uni3C attention имеет head_dim 64 — custom FA2 проходит probe только для 128, поэтому эти вызовы идут через SDPA (с `strict_attention=true` будет ошибка).
* Animate: motion encoder лица пересчитывается на каждом шаге (как в native) и на каждом ранке; для длинных видео это заметная доля времени шага.
* Animate2 cache: перед заполнением проверяется RAM — кэш всех rank должен занимать ≤60% `MemAvailable` (400+ кадров 960×544 ≈ 14 GB на rank, ~84 GB на 6 GPU в fp16 — плюс 33 GB pinned весов не помещается в 128 GB). Если не помещается, шаги идут без кэша с предупреждением.
* weight_placement=ats: веса в managed memory, страницы мигрируют между RAM и GPU по обращению; когда активации большие (длинные видео), драйвер вытесняет страницы весов и каждый блок заново подкачивает их — это и выглядит как «постоянный своп». Для Wan на V100 используйте `cpu` (явный FSDP CPU offload по NVLink), ats — экспериментальный режим.
* Animate2 без cache: pose branch почти удваивает вычисления шага (как в native). Память Animate2 в token-режиме: полный K/V генерации + pose на каждом ранке — для 720p используйте Ulysses.
* LoRA только на fp16/bf16/fp8 чекпоинтах (не в INT8). Изменение LoRA или опций перезапускает workers.
* `both` + placement=gpu на 720p, скорее всего, не поместится в 16 GB. Оценки памяти — оценки, не гарантия.
* При одном только high-проходе (без последующего low) workers остаются активными до следующей задачи/выгрузки моделью ComfyUI или `PowerShardRelease`.

## Добавленные файлы

`powershard/wan_config.py`, `wan_lora.py`, `wan_model.py`, `wan_extras.py`, `wan_text.py`, `wan_free.py`, `wan_backend.py`, `wan_worker.py`, `wan_runtime.py`, `wan_adapter.py`, `wan_nodes.py`; `scripts/build_wan_workflows.py`, `scripts/probe_wan_cuda.py`; `tests/test_wan_*.py`, `tests/wan_*.py`; `workflows_wan/*`; `README_WAN22_RU.md`, `RESEARCH_WAN22_RU.md`; дописаны строки в конец `__init__.py`.
