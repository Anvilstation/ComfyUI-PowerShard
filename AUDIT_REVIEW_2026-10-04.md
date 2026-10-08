# Аудит ComfyUI-PowerShard 0.5.0 — 2026-10-04

Проверен предоставленный исходный архив и лог `Text Document.txt`. Исправления внесены в код, UI, workflows и диагностические скрипты. Целевой сервер: IBM AC922 / POWER9 ppc64le, 6×Tesla V100 16 GB, RAM 128 GB, Ubuntu 20.04.5, custom PyTorch 2.12 / CUDA 12.4 / FA2. Доступная проверочная среда — x86_64 CPU; CUDA/AC922 не доступны.

## Что установлено по логу

В логе две разные проблемы. Импорт запрошенного `flash_attn` падает с `ModuleNotFoundError: No module named 'flash_attn_2_cuda'` на четырёх выбранных GPU. После этого выбирается SDPA. Позже finite tracker сообщает non-finite только на output. Импорт FA2 не является доказанной причиной последующего NaN.

Прежняя диагностика утверждала, что промежуточные tensors конечны, даже когда глубокая проверка выключена. Поэтому точное место появления NaN по этому логу определить нельзя. В коде найдены два опасных сужения FP32→FP16 в sequence path, воспроизводящие такой класс ошибок. Они исправлены и покрыты регрессиями. Окончательное подтверждение причины **конкретного серверного падения** требует повторения исходного workflow с debug finite на AC922.

## Подтверждённые дефекты и правки

| Область | Дефект исходного кода | Исправление |
|---|---|---|
| Sequence token KV | Восстановленный FP32 V сужался до FP16 до вычисления масштаба; значения >65504 становились Inf | Safe path сохраняет FP32 при gather; FP16 для kernel появляется после scaling |
| Final sequence assembly | FP32 residual сужался до FP16 перед финальным gather | Итоговый stream собирается в исходном dtype |
| Ulysses layout | BLHD output dispatcher передавался в обмен, ожидавший LHD; лишняя batch-ось маскировалась broadcasting | Batch=1 удаляется явно до обратного all-to-all |
| Число GPU | Требовалась делимость heads на world, иначе режим менялся | Padding heads и точная обратная перестановка для произвольного world |
| Gather ownership/VRAM | Глобальный cache сохранял разные video shapes; результаты делили изменяемую память | Каждый result владеет отдельным allocation, временные буферы освобождаются |
| ATS | ATS был CPUOffloadPolicy с late UVA/copy probe; Driver API использовал enum 27 | Отдельный scoped managed allocator, проверка Driver attrs 41/83/88/89/100 до загрузки |
| Default Comfy allocator | cudaMallocAsync несовместим со scoped MemPool | Native allocator выбирается только в ATS subprocess; parent Comfy не меняется |
| CPU lifecycle | empty_cache вызывался между denoising steps; staged conditioning мог накапливаться у kept workers | Reuse временных allocations между шагами, release/cache clear на границе run |
| Finite diagnostics | Output-only тест объявлял intermediates конечными | Честное сообщение, первый наблюдавшийся неисправный модуль при deep mode, один отложенный D2H read |
| MLP | Chunking зависел от numerical patch; planner не учитывал полный result | Независимые H3/Qwen MLP-ноды, полный output вычитается из бюджета |
| Empty INT8 shards | ConvRot reshape(0,-1,group_size) ломался на пустых последних shards | Явная обработка пустого tensor до dequantization |
| Wire protocol | Отсутствующий tensors file сдвигал `$dir` индексы staged sources | Слоты sources сохраняются, неверная ссылка выдаёт явную ошибку |
| Qwen memory selection | idle=cpu_shards мог молча заменить выбранный GPU/ATS режим на CPU | Несовместимая политика отклоняется; для gpu/ats выбрать release/keep |
| Cleanup | Ошибка end_run/report могла скрыть исходную sampling exception | Исходная ошибка сохраняется; secondary cleanup error отражается отдельно |
| Audit tooling | memory audit не запускал Session до after-load снимка, использовал неверные H3 shapes и host allocator stats | Явный session.start, корректные native audio/video shapes, статистика workers и driver по UUID |
| Acceptance tooling | Tiny H3 probe не включал sequence; compare_runs сравнивал общий label inputs | Реальный sequence в probe, наборы 3/4/5/6, hash фактического входа для сравнения |
| Workflows | Старый порядок widgets и встроенные MLP controls несовместимы с новым UI | Обновлены 24 API/UI пары; migration известных legacy formats сохраняет связи и настройки |
| NUMA | libnuma mask не освобождалась | Освобождение mask в finally |

FP16 Safe теперь включён в H3 Loader по умолчанию. Это mixed precision с FP32 stream и масштабируемыми FP16 GEMM; файл весов не переводится целиком в FP32. В отдельных workers запрещены reduced FP16 accumulation/reduction. Математический dense attention и INT8 ConvRot сохраняют прежний контракт.

## UI после изменений

Основная Config-нода содержит пять параметров: GPU IDs, weight placement, precision, attention backend, sequence mode. Backend закреплён на `fsdp2_sequence`; `fsdp2` исключён из UI, legacy named value нормализуется с warning. Таймаут генерации и allow_unverified исключены. У PyTorch NCCL остаётся внутренний watchdog, выставленный на 365 дней; пользовательского RPC deadline нет. Compatibility probes и остановка subprocess имеют отдельные технические пределы. Не нужно управлять memory_policy или communication dtype из основного UI: используется auto planner, safe sequence communication сохраняет FP32.

Дополнительная ConfigTuning содержит только работающие controls: reserve, prefetch, NUMA affinity/bind, keep workers, strict attention, host wrappers, pin memory. Она необязательна. Параметры MLP вынесены в `PowerShardH3MLP` и `PowerShardQwenMLP`; FP16 Patcher содержит safe/debug. В manual chunk_tokens задаёт лимит, auto рассчитывает его, off исполняет полный локальный MLP. Сторонние LoRA/ControlNet/weight-changing wrappers остаются вне поддерживаемого remote contract.

UI migration использует официальный beforeConfigureGraph hook и чистое JSON-преобразование на existing Comfy endpoint; веса и файлы этот endpoint не изменяет. Поддержаны известные legacy Config layouts на 7, 14 и 15–18 widgets, H3 Patcher и Qwen Loader. Неизвестные layouts и связанные legacy advanced controls требуют named API migration/ручной настройки. Browser integration на реальном сервере не запускалась. Обоснование hook: [официальный ComfyUI frontend interface](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/types/comfy.ts).

## Как различаются три режима

| Режим | Постоянные shards | Что занимает обычную VRAM |
|---|---|---|
| gpu | CUDA allocations | Shards, текущие FSDP groups, activations, communication, workspace |
| cpu | RAM; pinned configurable | Текущие собранные FSDP groups, activations, communication, workspace |
| ats | cuMemAllocManaged, CPU preferred, scoped pool | Текущие FSDP groups, activations, communication, workspace; managed residency контролирует CUDA |

ATS не включает глобальную подмену allocator процесса ComfyUI и не является повторным названием CPU offload. CUDA driver проверяется до heavy weights; каждый persistent parameter проверяется как managed pointer. Tiny preflight проверяет сохранение managed allocation после повторных FSDP forwards. C++ allocator собирается локально посредством g++/c++ и libdl, без nvcc/headers. Если аппаратных атрибутов ATS нет, запуск завершается с диагностикой.

Аппаратная поддержка Unified Memory на Power9+Volta описана NVIDIA; её уровень следует определять по capability attributes. UVA сама по себе недостаточна. Managed memory может превышать ёмкость VRAM, а физическое размещение страниц меняется независимо от виртуального указателя. Это основание реализации, а не результат измерения на данном AC922. [CUDA 12.4.1 Programming Guide, Unified Memory](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#unified-memory-programming).

RAM 128 GB — общий ресурс сервера. Pinned weights, pageable/file-backed pages, encoder, VAE, staging и runtime overhead используют его совместно. Стартовые graphs используют INT8 H3/Qwen и release Qwen перед H3. BF16 Qwen в полном CPU FP32 может занять значительную часть RAM; такие графы не выбраны стартовыми.

Managed allocator касается persistent weights. Он не устраняет OOM activations/active groups. Torch allocated/reserved для managed pool — логические размеры, поэтому Comfy budget для ATS использует обычные active-group estimates. Физическую GPU загрузку следует смотреть по driver/NVML, RAM — по worker RSS и системным метрикам. Даже сумма RSS процессов может повторно учитывать общие mmap pages.

## Поддержка произвольного world в Ulysses

Перед обменом H heads расширяются до H′=ceil(H/P)×P. Q/K/V упаковываются по destination rank, обмен переводит token shards в head shards. Нулевые token tails обрезаются **до** attention, чтобы не менять softmax denominator. После attention выполняется обратная перестановка и отрезаются искусственные heads. Residual сохраняет FP32.

Для H3 с 56 heads:

| GPU | Heads после padding | Heads на rank | Дополнительные heads |
|---|---:|---:|---:|
| 3 | 57 | 19 | 1.79% |
| 4 | 56 | 14 | 0% |
| 5 | 60 | 12 | 7.14% |
| 6 | 60 | 10 | 7.14% |

Эти проценты описывают padding heads, не время всего H3. Custom FA2 probe проверяет полные и локальные heads на каждой карте. Полное число heads определяется checkpoint. У Qwen свой token/GQA path; Ulysses-переключатель H3 не означает head sharding Qwen vision.

Если torch.chunk создаёт пустой последний token rank, включается явно отмеченный replicated compute с FSDP weights. Карты не исключаются. Tiny tests/probes используют достаточно токенов, чтобы проверить действительный sequence path на 3/4/5/6 GPU, а не этот fallback.

## Выполненная проверка

| Проверка | Результат и границы |
|---|---|
| Полная pytest suite с native ComfyUI | **219 passed, 16 skipped**, 22.86 s; x86_64 CPU |
| Native H3 / ComfyUI sampler / subprocess wire | PASS; tiny random weights, audio+video, clones, conditioning, cancellation/error paths, repeated seed |
| Ulysses tensor exchanges | PASS для world 3/4/5/6/9, heads 1/7/56, неравномерных токенов, V порядка 10⁶; transport эмулирован |
| Native H3 token/Ulysses + large residual | PASS на 3/5/6 логических ranks; native model, FP32 >65504, SDPA, emulated transport |
| Legacy/current workflows | PASS migration и проверка ссылок/отсутствия циклов всех 24 API/UI пар |
| CUDA Driver attribute test | PASS mock: UVA без host page tables не принимается как ATS |
| C++ managed allocator compilation | PASS g++ на x86_64; загрузка libcuda/выделение памяти НЕ запускались |
| Static Python/syntax checks | PASS undefined names и compileall |
| Real Gloo collectives/native sequence | **NOT_RUN**: EPERM/Operation not permitted при создании TCP transport; 4+12 тестов пропущены |
| CUDA/NCCL FSDP2 и custom FA2 | **NOT_RUN**: CUDA отсутствует |
| ATS / POWER9 / real checkpoint / visual quality / speed | **NOT_RUN** |

ComfyUI test commit: `f1072eb0350638a3390ddb6afbcaa8c6b237c6fd`. Test runtime: Python 3.12, torch 2.12.0+cpu; отдельно от пользовательского CUDA окружения. Текущий source admission основан на capabilities/signatures, не на запрете неизвестного SHA. Первоначальные reports сохранены как исторические и не являются результатами этой версии. JUnit и log данного запуска: `reports/review-2026-10-04/`.

Факт успешного CPU/layout теста не подтверждает NCCL, sm_70 kernel, pinned CPU Offload или oversubscription на AC922. Выше приведена точная граница проверки; цифры GPU speedup отсутствуют.

## Research и предлагаемые следующие доработки

1. **Сначала проверить реальный custom FA2 provider.** В вашем логе FA2 фактически не использовался. Auto теперь проверяет оба поддерживаемых интерфейса прежде SDPA. После успешного isolated probe сравнить native H3 outputs и memory peak на одинаковых seed/prompt. SDPA является semantic baseline, конкретный CUDA kernel определяется dispatch. Не устанавливать обычный wheel поверх ppc64le custom stack.
2. **Сравнить token и Ulysses на всех шести картах.** All-to-all head sharding снижает необходимость полного KV replication; преимущество на конкретной AC922 NVLink/NUMA topology нужно измерить. Основание конструкции: [оригинальная DeepSpeed Ulysses работа](https://arxiv.org/abs/2309.14509). Результаты её обучения на других GPU не являются ожидаемым speedup H3/V100.
3. **Prefetch 0→1 только после измерения пика памяти.** CPUOffloadPolicy возвращает веса на device перед all-gather; pinned memory помогает асинхронным передачам. Prefetch может перекрывать communication/compute, повышая VRAM cost. Рекомендация — сначала 0, затем парный тест 1; 2 оставить при достаточном резерве. Основание: [FSDP2/CPUOffloadPolicy документация PyTorch 2.12](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html).
4. **NUMA auto и измерение H2D/D2H.** Сравнить выбранный UUID/rank order, CPU affinity, RAM locality и реальные transfer traces. Для cpu/ats использовать тот же GPU subset и inputs. Bind требует numactl и корректной topology; автоматически угадывать единственный NUMA узел для всего сервера нельзя.
5. **Подобрать MLP chunks по wall time и peak.** Сравнить 1024/4096/8192/16384/auto/off при неизменном FP16 Safe. Большие chunks уменьшают число launches, но повышают workspace; на INT8 отдельно смотреть стоимость dequantization/prepared matrices. Не отключать FP32 residual ради размера chunk.
6. **ATS оставить экспериментальным до сравнения с explicit CPU offload.** PyTorch поддерживает scoped MemPool для custom allocators; UVM page faults и эвикции могут ухудшать производительность. Если задача помещается в VRAM, explicit placement обычно предпочтительнее. Это вывод из документации, не измерение AC922. [PyTorch 2.12 CUDA semantics, custom allocator/UVM](https://docs.pytorch.org/docs/2.12/notes/cuda.html#using-custom-memory-allocators-for-cuda).
7. **Дополнить UI фактическими метриками выполнения.** Компактный status panel: requested/effective attention, rank mapping, persistent vs active weights, actual CUDA peak/RAM, reason fallback и дельта между runs. Сейчас выбор GPU показывает доступность provider modules, а effective backend находится в worker logs. Panel должен брать worker reports, не делать вывод по импорту package.
8. **Оптимизировать FP32 communication отдельным безопасным протоколом.** Следующая возможность — scaled FP16 wire с коллективно согласованным масштабом и компенсацией. Нужны high-range/uneven-head regressions и сравнение latents: простой `.half()` возвращает найденный баг. До такой проверки сохранён FP32.
9. **Перейти от оценок activations к измеряемому autotuning.** Planner не знает всего reference/keyframe/vision expansion и fragmentation. Практическая доработка — cache безопасных chunks/prefetch по геометрии, placement, provider и GPU UUID с измеренным peak; память result и active groups учитывать отдельно. Не превращать heuristic в обещание, что OOM невозможен.
10. **Ограничить длительное хранение telemetry history и проверить IPC стоимость.** Kept Session history пока сохраняется между runs; это диагностические JSON, не веса, но при длительном сервисе может расти. Выгружать завершённые run reports и обрезать history после сохранения. File-based IPC/staging также измерять отдельно от kernel; shared-memory транспорт — будущая доработка после проверки cancellation и ownership.

Это список предлагаемых доработок. Status panel, scaled FP16 wire, autotuning, ring attention и новый IPC в эту версию не добавлены.

## Воспроизводимая серверная приёмка

Сначала tiny NCCL FSDP2 probe из README на 3/4/5/6 GPU. Для каждого набора повторить token/Ulysses, gpu/cpu и, после capability gate, ats. Сохранять разные output paths. Затем реальный H3 `accept_h3.py` с --debug-finite и --lifecycle. Этот тест синтетический и не заменяет generation quality.

Для настоящей генерации открыть `ac922_6gpu_cpu_ulysses.ui.json`, выбрать свои файлы и prompt, запустить 512×512/25 frames/8 steps, сохранить latents/PNG и logs. Затем сравнить с token и GPU при одинаковом seed и без Spectrum. Spectrum является приближённым ускорением: его качество проверять отдельным сравнением. При сравнении full execution через Comfy cache использовать --cache-none либо убедиться, что sampler действительно исполнялся.

Для выделения одной оси:

```bash
python scripts/benchmark_cases.py workflows/ac922_6gpu_cpu_ulysses.api.json --axis sequence --output-dir reports/local-sequence-matrix
python scripts/benchmark_cases.py workflows/ac922_6gpu_cpu_ulysses.api.json --axis mlp --output-dir reports/local-mlp-matrix
python scripts/benchmark_cases.py workflows/ac922_6gpu_cpu_ulysses.api.json --axis placement --output-dir reports/local-placement-matrix
python scripts/benchmark_attention.py --gpus all --backend auto --heads 10 --head-dim 128 --length 256 --output reports/local-attention.json
python scripts/benchmark_transfer.py --gpus all --numa-policy auto --output reports/local-transfer.json
```

Matrix scripts без --submit только создают графы. При переключении placement сохранять Qwen idle policy release или отдельный CPU config для cpu_shards. Отчёты CUDA peak включают allocations/communication/dequant, а logical communication bytes не являются измеренным NVLink traffic. Учитывать cold load, warm forward, encoder cache, VAE и codec как разные стадии.

В результате поставляется исправленный код и проверяемые инструменты приёмки. Утверждение о закрытии конкретного CUDA падения, устойчивости ATS и ускорении всего H3 можно сделать только по серверным результатам.
