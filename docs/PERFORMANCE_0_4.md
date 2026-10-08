> Историческая документация до 0.5.0. Старые UI/CLI аргументы и результаты не относятся к текущей ревизии. Актуальные команды: [README_RU.md](../README_RU.md), результаты: [AUDIT_REVIEW_2026-10-04.md](../AUDIT_REVIEW_2026-10-04.md).

# PowerShard 0.4: MLP, память, Qwen и Spectrum

Продолжена база `f4a53c8a2e17c4f01741e65ef46deffc36531071` (0.3.0), без переписывания runtime или ComfyUI core. Новые настройки добавлены после прежних widgets. Старые JSON workflows не изменились. Python 3.11 и установленный серверный torch/CUDA/NCCL/custom wheel заменять не нужно.

## Ответы на наблюдения AC922

**48 секунд FSDP против 2–3 минут sequence означает замедление sequence в 2.5–3.75 раза.** Это USER_REPORTED, не наш benchmark: нет совпадающих workflow/logs для атрибуции. В присланных MP4 отсутствуют seed, prompt, sampler, backend и generation latency. Из 7 файлов 5 уникальных, разрешение 960×544, длительности 1.625 и 5.166667 секунды — это длительности контейнеров, не время вычислений. Полный разбор: `reports/video-metadata-20260916.json`.

В FSDP-only каждый rank вычисляет полный пример с временно собранным активным блоком. В sequence каждый rank вычисляет свои queries и MLP tokens, но получает глобальные K/V на каждом DiT-блоке; в конце собираются hidden states. Heads не делятся между ranks, ограничения heads % world_size нет. На коротких последовательностях обмены/упаковка и маленькие GEMM могут оказаться дороже сэкономленных FLOPs. Копии global K/V и FP32 residual особенно заметны на длинных видео. Полную причину разницы на AC922 установит trace, не факт загрузки всех GPU.

Подтверждённые изменения кода:

- K/V объединены в один tensor/collective на блок вместо двух. Объём данных не стал вдвое меньше; уменьшилось число запусков обмена. Padding удаляется до attention.
- INT8 MLP может один раз подготовить scaled-half tiles активного MLP и использовать их для всех token chunks. После MLP tiles удаляются, постоянные параметры остаются INT8 + FP32 scales. При недостаточном бюджете остаётся прежняя ограниченная деквантизация строк.
- Для dense MLP подготовка scale/bounds матриц также переиспользуется внутри одного MLP. Нет cache весов между diffusion steps.
- Повторный обход metadata всех установленных Python distributions убран из каждого RPC fingerprint. На здешнем CPU median warm `provider_stamp`: 199.03 мс → 0.765 мс. Это измерение этой функции, не ускорение H3 или NVLink.

## MLP chunk

`mlp_chunk_tokens` делит только token rows MLP. Это не разбиение attention, не число diffusion steps и не FSDP group size. FSDP оборачивает весь DiT-блок: внутри MLP chunk loop нет дополнительных FSDP collectives. Малые порции уменьшают активации, но создают больше GEMM и, без подготовки весов, больше деквантизаций. Поэтому увеличение до 16384 могло ускорить ваш запуск.

| Режим FP16 Patcher | Поведение |
|---|---|
| manual | Указанный размер, по умолчанию 512 для старых графов. Искусственного потолка 16384 нет. |
| off | Весь локальный MLP одним вызовом по tokens. FP16 Safe остаётся включённым. |
| auto | Размер из локального числа tokens и оценки оставшегося бюджета; решение записывается в metrics. |

Значение 0 не означает «отключить»: используйте **mlp_chunk_mode=off**. Большой/off chunk может получить настоящий OOM; прогноз памяти не запрещает попытку. Auto не меняет seed/разрешение/длительность/steps и не включает CPU offload скрытно. Сравнивайте off/manual/auto с одинаковыми inputs. CPU regression охватывает 1/4/1024/4096/8192/16384 tokens, INT8 и dense, finite и численное отличие.

## Что переносит offload

`cpu_offload=true` использует штатный FSDP2 `CPUOffloadPolicy(pin_memory=...)`: **локальные шарды** хранятся на CPU, нужная группа переносится на GPU, собирается и после использования снова шардируется. CPU offload дополняет FSDP. Meta-init и `safe_open.get_slice()[local_rows]` не создают N полных CPU моделей. Реальные PSS/pinned/loading peaks на AC922 ещё не измерены.

«Слои активации» как отдельные постоянные слои модели отсутствуют. В inference без backward освобождённые промежуточные tensors повторно не нужны. Текущие residual/KV/workspace всё равно должны поместиться; offload весов не даёт верхнюю границу VRAM. Суммарные 48 ГБ трёх карт также не являются единой 48-ГБ allocation: каждый rank ограничен своей свободной памятью.

`memory_policy=auto` считает свободную память на границе RPC, вычитает reserve, оценку активаций/коммуникаций и active groups, уменьшает prefetch до 0..заданного, согласует минимум бюджетов ranks и отдаёт остаток MLP. На длинных запросах сначала уменьшается лишний prefetch; chunk_auto использует остаток. Это консервативные оценки, не гарантия размещения. Для Qwen image/video expansion до native processor неизвестно, поэтому оценка помечена lower bound. Автоматического retry после CUDA OOM/illegal access нет: session завершается.

Prefetch использует `set_modules_to_forward_prefetch`, глубина 0/1/2. Наличие API не доказывает overlap CPU↔GPU на конкретном AC922. `numa_policy=auto` использует найденную locality для CPU affinity; `bind` применяет numactl, если доступен. Нельзя считать каждый transfer NVLink-передачей без topology и измерения. `benchmark_transfer.py` сравнивает pageable/pinned H2D/D2H, никаких глобальных NCCL_P2P_DISABLE по умолчанию.

Единственный новый выборочный offload долгоживущих активаций — **Spectrum history на CPU**. Snapshot владеет своей памятью; перенос обратно идёт порциями, без N полных историй. Это синхронный корректный путь, без обещания перекрытия. Общий activation-offload и autograd saved_tensors_hooks не реализованы и не объявляются решением inference.

## H3 Qwen Loader

Нода `PowerShard H3 Qwen Loader` выдаёт настоящий наследник `comfy.sd.CLIP`, native tokenizer/template/vision/DeepStack и `minimax_token_tags`. На audited checkpoint: vocab 151936, hidden 5120, intermediate 25600, 50 decoder layers, Q heads 64, KV heads 8, dim 128. Это подтверждено всеми 902 именами/формами BF16 header на полной meta-модели. Извлекается последний **ненормализованный** hidden state truncated H3 encoder; pooled embedding и lm_head не подставляются.

Поддержаны native text, image и 2-frame reference-video blocks; audio reference даёт текстовую метку H3, waveform в Qwen не подаётся. Малые настоящие ComfyUI классы проверены для text/image/video. Качество полного pretrained 32B encoder ещё NOT_RUN.

FP16: источник BF16 читается локальными строками, storage floating weights → FP16 с finite/range проверкой. Тяжёлые Linear используют scaled half GEMM и возвращают FP32; residual/norm/SiLU и небольшая vision patch Conv3d — FP32. Это отдельная Qwen strategy, не blind H3 DiT patch. INT8: существующий checkpoint ConvRot, integer weights и scales реально являются FSDP parameters. Активные строки/MLP временно деквантуются; вся модель dense не разворачивается. BF16→INT8 конвертер не добавлен.

FSDP units: decoder blocks, embedding, vision blocks, patch/mergers, visual root для напрямую используемого pos_embed, root. Все reshard_after_forward=True; explicit reshard/assert после RPC, no_grad/inference_mode(False), optimizer отсутствует. Генерируемый rotary buffer восстанавливается после to_empty. Свой pipeline/runtime не создавался: новая роль `qwen` в прежнем Session/worker.

`fsdp2_sequence` для Qwen разделяет queries/MLP rows, собирает глобальные KV и hidden после **каждого** decoder layer для сохранения native DeepStack. Vision compute реплицирован при шардированных весах. Это реализация распределённых вычислений в коде, пока не измеренное ускорение. Native text и multimodal RoPE layouts, causal rows и GQA проверены на CPU с подставленными collective payloads; это не доказательство NCCL. Слишком короткий запрос использует FSDP-only на всех выбранных ranks.

Attention selector передаётся в encoder workers. Непроверенный GQA/custom-Flash контракт использует SDPA/math fallback; vision остаётся FP32 SDPA/math. Отчёт показывает `qwen_text`/`qwen_vision` call_counts и причину fallback. Поэтому выбор vllm-fa не означает, что весь Qwen вычислялся этим kernel. При allow_fallback=false неподдерживаемый контракт завершает запуск понятной ошибкой.

| idle_policy | После conditioning |
|---|---|
| release (default) | Workers закрываются; bounded CPU conditioning cache остаётся. |
| cpu_shards | Workers/NCCL context сохраняются, shards остаются на CPU, allocator освобождается на границе фазы. Если offload был false, явное warning показывает effective true. |
| keep | Можно переиспользовать encoder session; перед активной DiT фазой она освобождается. Две большие GPU-роли не держатся одновременно. |

CPU idle context всё ещё занимает немного VRAM/NCCL/buffers, это не «0 GPU bytes». Release node по умолчанию сохраняет idle Qwen CPU shards; выключите preserve_qwen_cpu_shards для полного завершения. Параметр clear_conditioning_cache очищает bounded caches. Cache key включает содержимое tokens/images, tokenizer hash, checkpoint path/size/mtime, precision/provider/config/options. Возвращаются CPU copies, clone не разделяет изменяемый cache/options. Полный SHA многогигабайтного checkpoint при каждом RPC не вычисляется; файл, изменённый с сохранением size/mtime, надо явно переименовать/перезагрузить.

## Spectrum

Изучен [xmarre/ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) commit `120d72e2f48b781235b34149e39bbdf0f1317d82`, GPL-3.0-or-later. Upstream оставляет multi-GPU sampling native: distributed forecast transactions не валидированы. Наш адаптер реализует собственную согласованную worker-транзакцию, без global monkey patch или копирования solver bridges.

`PowerShard Spectrum` MODEL→MODEL **выключен по умолчанию**. Host clone передаёт JSON policy в каждую session. После wrapping/loading FSDP ставятся внешние gate modules: на ACTUAL они вызывают исходный FSDP child; на FORECAST не вызывают его __call__/all-gather вообще. Перед root forward все ranks голосуют readiness, MIN + rank0 broadcast дают единое решение. Локальный runtime fallback посреди FSDP не выполняется.

История — только target audio/video hidden после последнего DiT-блока, разделённая по rows между ranks, даже в FSDP-only. Text/reference tokens не сохраняются в истории. Прогноз — Chebyshev basis + ridge + линейная экстраполяция; native текущие input/time projections и final video/audio heads продолжают выполняться. Audio spectral blend по умолчанию 0 (линейная часть); video blend .5. Наши default degree=2/ridge=.001/warmup=3 — явная экспериментальная policy, не воспроизведение всех настроек/качества upstream. max_forecast=1, затем actual refresh, tail>=1.

Поддержан bridge native deterministic Euler с s_churn=0. Heun/ER-SDE/SEEDS/SA/чужие sampler functions дают warning и **обычный FSDP**, без подмены solver history. Нет admission по GPU/версии. Run ID, seed/noise boundary, conditioning content, shapes, branch и timestep разделяют историю. Повторный/nonmonotonic timestep сбрасывает anchor lane. Clone/отключение ноды не меняют исходный MODEL. end_run/ошибка/новая session очищают историю.

Budget history_mib действует на все branches каждого rank; history_size ограничивает anchors, вспомогательный LRU — 256 lanes. На FORECAST собирается только один текущий target tensor, не вся история. CPU history переносится по 16 MiB; actual snapshot ownership и отсутствие alias проверены. Недостаточный budget означает ACTUAL, а не потерю математики текущего native forward.

В тесте native Euler 7 steps появляются реальные forecast-вызовы внутри отдельного CPU worker; DiT child prehooks не вызываются на skip. При 4 шагах warmup+tail дают 0 forecasts — это корректно и **не ускорение**. Реальная audio/video fidelity, Turbo-weights, CUDA/FSDP gates/collectives, combined sequence+Spectrum — NOT_RUN. Не следует включать приближённый режим в контрольный baseline.

## Замеры на сервере

Из каталога проекта, Python активной ComfyUI:

```bash
PS_PYTHON=/ABS/venv/bin/python
PS_COMFY=/ABS/ComfyUI
"$PS_PYTHON" scripts/diagnose.py --comfy "$PS_COMFY" --output reports/ac922-environment.json
"$PS_PYTHON" scripts/probe_devices.py --gpus all --timeout 180
"$PS_PYTHON" scripts/probe_h3_cuda.py --comfy "$PS_COMFY" --gpus 0,1,2 --cpu-offload
"$PS_PYTHON" scripts/benchmark_transfer.py --gpus all --numa-policy auto --output reports/ac922-transfers.json
"$PS_PYTHON" scripts/accept_qwen.py --comfy "$PS_COMFY" --checkpoint "$PS_COMFY/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" --gpus 0,1,2 --cpu-offload --idle-policy cpu_shards --output reports/ac922-qwen
POWERSHARD_BENCHMARK_DIR="$PS_COMFY/output/powershard" bash scripts/launch_comfy.sh "$PS_PYTHON" "$PS_COMFY" --cache-none
```

N=3 здесь только пример; 5,2,0 и all сохраняют выбранный порядок. Для sequence encoder добавьте --backend fsdp2_sequence. Для H3 реальные веса: прежний `accept_h3.py --lifecycle`; синтетический conditioning не считать полной генерацией. Для Qwen native численного сравнения передайте --reference conditioning.safetensors, созданный из того же native encoder/input. Допуски CLI взяты из tiny CPU проверки; полноразмерный контроль может потребовать отдельно обоснованных допусков.

Откройте `workflows/fl2va_qwen_int8_offload.ui.json`, выберите реальные файлы; оставьте Spectrum вне первого запуска. Для 15 секунд — `fl2va_long_auto.ui.json` (361 frames при 24 fps; native rounding/VAE определяют фактическую длительность), без гарантии размещения. Все новые graphs сохраняют PNG после VAE до video encoder.

```bash
"$PS_PYTHON" scripts/benchmark_cases.py workflows/fl2va_qwen_int8_offload.api.json --axis backend --output-dir reports/compare-backend
"$PS_PYTHON" scripts/benchmark_cases.py workflows/fl2va_qwen_int8_offload.api.json --axis mlp --submit --output-dir reports/compare-mlp
```

Без --submit это явное сохранение воспроизводимых inputs/плана. Оси backend/mlp/offload/attention/spectrum меняются по одной, seed/steps/prompt одинаковы. При --submit нужны idle локальный ComfyUI и реальные checkpoints. Проверяйте execution_cached: cached sampler не является benchmark. --cache-none может пересоздавать loaders — cold/warm различать по фактическому load в worker history. Warm forwards/cache hits отдельно проверяются accept_h3/accept_qwen.

`output/powershard/run-<id>.json` создаётся на каждом PowerShard sampler run: seed/noise hash, sampler/sigmas/latent shapes/config, per-rank history и metrics. Включённый POWERSHARD_BENCHMARK_DIR добавляет full graph/node stage report с text/conditioning/VAE/save timings. Raw prompt по умолчанию хешируется, tensors не сохраняются. `POWERSHARD_DEBUG_DUMP_INPUTS=1` разрешает prompt в benchmark report, но не выгрузку всех tensors. Worker reports содержат local shard bytes, allocated/reserved/peaks, PSS/RSS, effective attention calls, MLP plan, sequence exchanges и Spectrum decisions. Неточное разложение activation/workspace помечено NOT_SEPARATELY_ATTRIBUTED.

Для одного диагностического прогона задайте `POWERSHARD_PROFILE=1`: первые два RPC каждого worker сохранят PyTorch trace с shapes/memory и collective events. Это добавляет overhead; не смешивайте trace latency с обычным benchmark. Вложенные/overlapped CUDA events нельзя складывать с wall time. Стандартные операции не получают постоянных sync в каждом блоке.

Приоритет ускорений для A/B проверки: MLP manual 4096/8192/16384/off; выбор уже работающего attention; prefetch 0/1 при достаточной памяти; полное освобождение Qwen GPU перед denoising; NUMA placement по измерению; затем sequence и Spectrum по отдельности. Больше GPU/offload/prefetch не гарантируют меньшую latency. Уменьшение steps/разрешения/кадров меняет задачу и не считается той же оптимизацией.
