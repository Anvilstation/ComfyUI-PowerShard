# PowerShard LTX-2 / 2.3 / 2.5 + ускорители — дополнение к ComfyUI-PowerShard 0.5.3

Распределённый LTX (аудио+видео LTXAV и видео-only LTX-Video) на нескольких V100 (IBM AC922, ppc64le),
распределённый Gemma-энкодер и ускорители для Wan и LTX. Исходный код PowerShard не изменён: только новые
файлы и строки, дописанные в конец `__init__.py`.

## Ноды

Категория **PowerShard/LTX**:

| Нода | Что делает |
|---|---|
| `PowerShard LTX-2/2.5: Sequence Loader` | Полный checkpoint (`checkpoints/…`) или только трансформер (`diffusion_models/…`), bf16/fp16/fp8(_scaled). FSDP2 по GPU из `PowerShardConfig` + sequence parallel по видео-токенам. Выход — обычный `MODEL`. |
| `PowerShard LTX: численная политика / память` | fp16_safe, chunk FFN, batch_chunk, attention_chunk (v2a). |
| `PowerShard LTX: LoRA / IC-LoRA` | LoRA, IC-LoRA (union control, detailer…), distilled LoRA — сливаются в локальные строки shards при загрузке. Цепочкой. |
| `PowerShard LTX: Gemma Distributed Encoder` | Аналог `LTXAVTextEncoderLoader`: Gemma 3 12B / Gemma 4 + text projection шардируются по GPU (FP16 хранение, FP32 вычисление). Одинаковый текст — из кэша без запуска workers. |
| `PowerShard LTX: Video+Audio VAE из checkpoint` | Видео-VAE и аудио-VAE+vocoder из полного checkpoint, **не читая** 40+ GB весов трансформера (родной `LTXVAudioVAELoader` читает весь файл). |
| `PowerShard LTX: checkpoint info` | Геометрия/формат/оценка памяти без загрузки. |

Категория **PowerShard/Accelerators** (работают и для Wan, и для LTX):

| Нода | Что делает |
|---|---|
| `PowerShard: Block Cache` | TeaCache/FBCache-подобный пропуск блоков **внутри workers**: блок 0 считается всегда, блоки 1..N пропускаются, если относительное изменение residual блока 0 меньше порога. Решение — по all-reduce сумм, одинаковое на всех GPU. Кэш residual живёт в VRAM workers и очищается в конце задачи. |
| `PowerShard: NAG` | Normalized Attention Guidance (негативный prompt без CFG — для distilled/Lightning/LTX distilled при cfg=1). Wan: текстовая cross-attention блоков; LTX: видео и (опционально) аудио cross-attention. |
| `PowerShard Wan: RIFLEx` | Видео длиннее обучающих 81 кадра без «повтора» движения: одна временная частота RoPE Wan → 0.9·2π/L. k=0 — авто. |

Родные ноды сообщества/ComfyUI, которые работают с PowerShard-моделями **без** `allow_host_wrappers`:

* **Context Windows** (`ContextWindowsManual`, `WanContextWindowsManual`) — длинное видео окнами; для LTX-AV
  аудио режется синхронно (родной `resize_cond_for_context_window`). Pose-кэш Animate2 в окнах отключается,
  как у native.
* **EasyCache / LazyCache** — их diffusion-wrapper исполняется на host вокруг вызова workers (как в native forward).
* **LTXV STG** (`LTXVSpatioTemporalGuidance`), **Modality Guidance**, **LTXVReferenceAudio** (ID-LoRA),
  **LTXVDualCFGGuider** — post-CFG/guider логика на host; флаги прохода (STG-блоки, a2v/v2a) уходят в workers.
* **Guides / keyframes / IC-LoRA** (`LTXVAddGuide`, `LTXVAddLatentGuide`, `LTXVCropGuides`, generated keyframes,
  `LTXVImgToVideoInplace`), **латентный апскейлер** (`LTXVLatentUpsampler`), **Duration head**
  (`LTXVDurationPredictor` — коннекторы считаются в workers), **ModelSamplingLTXV / LTXVScheduler**.

Не переносятся в workers (явная ошибка с подсказкой): сторонние патчи `forward`/`forward_orig`
(TeaCache/MagCache от сообщества, KJ SageAttention-патч — используйте `Block Cache` и `attention_backend`
в `PowerShardConfig`), `patches_replace["dit"]`, native LoRA loader (используйте PowerShard LTX LoRA),
`GetICLoRAParameters` (читает patches LoRA; IC-LoRA с `reference_downscale_factor=1` работают без него).
FreeInit не добавлен: у flow-моделей (Wan/LTX) σ_max = 1 и «обратная диффузия» результата даёт тот же шум —
низкочастотная переинициализация не имеет смысла.

## Как устроено распределение LTX

* Видео-токены (десятки тысяч) режутся по rank до первого блока; **аудио-токены** (сотни) реплицированы.
* Видео self-attention — token (all-gather K/V) или Ulysses (all-to-all heads), как у Wan; с guide-маской
  (IC-LoRA/keyframes со strength ≠ 1) — token-путь с маской по глобальным строкам запросов.
* Текстовая cross-attention и **a2v** (видео-запросы → аудио K/V) — локально.
* **v2a** (аудио-запросы → все видео-ключи): частичная softmax по локальным ключам (online, FP32) +
  all-reduce MAX/SUM — точное значение полной attention, без сборки видео-токенов.
* AdaLN-модуляция: уникальные значения timestep → таблица, строки берутся индексом для локальных токенов
  (нет тензора `[B, T, 9·dim]`). RoPE считается только для локальных строк.
* Голова (`norm_out → proj_out`) — локально, собирается только `[B, T, 128]`.
* Численно: FP32 residual, AdaLN, RMSNorm, RoPE, timestep/caption/коннекторы; FP16 GEMM (опц. scaled Safe);
  FP16 attention со статическими степенями двойки для q/k. Без comfy-kitchen ядер (на ppc64le их нет).

## Готовые workflows (`workflows_ltx/`, API-формат)

`ltx2_t2v_audio`, `ltx2_i2v_audio`, `ltx2_keyframes_iclora` (первый+последний кадр, IC-LoRA),
`ltx2_two_stage_upscale` (640×384 → латентный x2 → короткий второй проход), `ltx2_long_video_context_windows`
(481 кадр окнами + Block Cache). Имена файлов моделей — плейсхолдеры (`ltx-2.3-22b-dev.safetensors`,
`gemma_3_12B_it.safetensors`, апскейлер); подставьте свои. Сгенерированы `scripts/build_ltx_workflows.py`.

Рекомендации для 6×V100 16 GB: `weight_placement=gpu` (22B fp16 ≈ 7.5 GB на GPU) при ≤ 121 кадре 768×512;
для длинных/крупных — `cpu` (pinned RAM) + `mlp_chunk_mode=auto`.

## Форматы весов и GPU: как грузить и как вычислять

Загрузчики Wan, LTX, MiniMax H3, umT5 и Gemma читают **любой формат ComfyUI**: bf16/fp16/fp32, fp8 (scaled и
`comfy_quant`), mxfp8, nvfp4, int8_tensorwise (в т.ч. ConvRot), int4 (convrot_w4a4, asym_w4a8_int8), w6a8 —
и новые форматы, которые понимает ваша ComfyUI (квантованный вес собирает родной код ComfyUI/comfy-kitchen).
Что делать с весами дальше, выбирается тремя параметрами:

| Параметр | Где | Значения |
|---|---|---|
| `weight_format` — **как хранить** Linear блоков в VRAM | `PowerShard Wan: Options`, `PowerShard LTX: Options`, Gemma-энкодер, `PowerShard MiniMax H3: Loader (форматы весов…)` | `dequantize` — fp16/bf16 (как раньше, по умолчанию); `as_file` — в формате файла без перевода (int8/int4/nvfp4/fp8 остаются как есть); `int8` / `fp8` / `mxfp8` / `nvfp4` / `int4` — квантовать при загрузке (bf16-файл → int8 и т.п.; если файл уже в этом формате — берётся как есть) |
| `compute` — **как считать** квантованные слои | там же | `auto` — native, если GPU и comfy-kitchen умеют формат, иначе деквантование; `native` — ядра comfy-kitchen/torch (int8/int4/fp8/fp4 GEMM); `dequantize` — вес на время GEMM переводится в fp16/bf16 (работает везде, VRAM экономится всё равно) |
| `weight_dtype` — dtype для неквантованных весов и GEMM | Wan/LTX Options | `auto` — bf16 на sm80+ (RTX 30/40/50, A100/H100/B200), fp16 на V100/T4; `fp16`; `bf16` |

Квантованное хранение шардируется FSDP так же, как fp16 (все части QuantizedTensor — данные, scales, codebook —
упакованы в один байтовый вектор), т.е. VRAM на GPU ≈ размер квантованного файла / число GPU.
FP32-острова (AdaLN, timestep, caption/коннекторы, головы) и слои с размерами не кратными блоку не квантуются.

Native-ядра по GPU (`compute=auto`; в отчёте загрузки `quant_report` видно, что выбрано и почему):

| Формат | Native нужно | V100 (sm70) | RTX 3090/A100 (sm80/86) | RTX 4090 (sm89) | RTX 5090 / B200 (sm120/100) |
|---|---|---|---|---|---|
| int8_tensorwise, int4 (convrot_w4a4, asym_w4a8, w6a8) | sm75+ и CUDA-backend comfy-kitchen | деквант | native* | native* | native* |
| fp8 (e4m3/e5m2) | sm89+ (torch scaled_mm) | деквант | деквант | native | native |
| mxfp8, nvfp4 | sm100+ и CUDA-backend comfy-kitchen | деквант | деквант | деквант | native* |

\* CUDA-backend comfy-kitchen требует torch cu130+; если его нет (как сейчас на AC922 с cu124), `auto` выбирает
деквантование, а `native` при ошибке ядра пишет предупреждение и переходит на деквантование для этого слоя.

Рекомендации:
* **6×V100 (AC922):** `weight_format=dequantize` (максимальная скорость) или `as_file`/`int8` + `compute=dequantize`,
  если не хватает VRAM (веса в 2–4 раза меньше, GEMM в fp16; цена — деквантование слоя на каждом шаге).
* **8×RTX 5090:** `weight_format=as_file` (nvfp4/fp8/int8-файлы) или `nvfp4`/`fp8`, `compute=auto`, `weight_dtype=auto` (bf16).
* `precision=int8_fp16` в `PowerShardConfig` по-прежнему включает старые Int8Linear-ядра Wan/H3, но только при
  `weight_format=dequantize` и int8_tensorwise-файле. При любом другом `weight_format` используется новый путь.
* MiniMax H3 с выбором формата — отдельная нода `PowerShard MiniMax H3: Loader (форматы весов / int8 / fp8 / nvfp4 / int4)`;
  исходный H3 loader не изменён.
* LoRA: при квантованном хранении LoRA вливается в вес и он переквантуется в ту же раскладку (ConvRot/группы сохраняются).
* ModelOpt-поля (`pre_quant_scale`, `input_scale`, `full_precision_matrix_mult`) учитываются, как в ComfyUI.
* FP16 Safe (V100) для квантованных слоёв: вычисление через деквантование + scaled half GEMM с FP32-выходом
  (native-ядра в этом режиме не используются — они дают fp16 выход без защиты от переполнения).
* GGUF (.gguf) не поддерживается — это не safetensors.

## Проверка на сервере

```bash
cd custom_nodes/ComfyUI-PowerShard
python -m pytest tests/test_ltx_config.py                                   # без torch/GPU
python -m pytest tests/test_ltx_native.py --comfy /path/to/ComfyUI          # эквивалентность с native LTXAV (CPU)
python -m pytest tests/test_quant_formats_torch.py --comfy /path/to/ComfyUI # деквантование + QuantLinear (pack/unpack)
python scripts/probe_ltx_cuda.py --comfy /path/to/ComfyUI --gpus all --sequence-mode ulysses
python scripts/probe_ltx_cuda.py --comfy /path/to/ComfyUI --gpus all --sequence-mode token --weight-placement cpu --fp8 --adaln
```
`probe_ltx_cuda.py` сравнивает с FP32 native: обычный шаг, коннекторы, STG, a2v/v2a off, guides с маской
внимания, Block Cache (второй вызов должен пропустить блоки: `block_cache_second_call_skipped=true`).

## Ограничения

* Код LTX/Gemma/ускорителей/квантованного хранения **не запускался на torch/GPU** в среде разработки (там нет torch); проверены
  заголовки/конфиги/workflows и математика v2a-комбинации (numpy). Перед работой прогоните тесты выше.
* Block Cache — эвристика (как TeaCache): порог 0.05–0.12; при артефактах уменьшите порог/`max_consecutive_skips`.
* NAG с `batch_chunk>0` и Block Cache учитывают части batch; Animate2 (отдельный pose-forward) — без них.
* Gemma: генерация текста (TextGenerate / prompt enhancer) через distributed encoder не поддерживается.

## Добавленные файлы

`powershard/ltx_config.py`, `ltx_model.py`, `ltx_backend.py`, `ltx_adapter.py`, `ltx_nodes.py`,
`native_te.py`, `accel.py`, `quant_formats.py`, `quant_linear.py`, `h3_quant.py`; правки собственных `wan_*` файлов (роль LTX/TE в session и worker, Block Cache /
NAG / RIFLEx / EasyCache / context windows для Wan); `tests/test_ltx_config.py`, `tests/test_ltx_native.py`,
`scripts/build_ltx_workflows.py`, `scripts/probe_ltx_cuda.py`, `workflows_ltx/*.api.json`.
