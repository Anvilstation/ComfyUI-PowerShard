# PowerShard 0.5.3: скорость, idle RAM и prefetch

Сервер пользователя: AC922 POWER9 ppc64le, V100 16 GB, 128 GB RAM; новые замеры сделаны на пяти картах. Пользователь сообщил 79–82 s/iteration SDPA token/Ulysses и 84 s/iteration ATS+token. Нового workflow, rank history или CUDA trace к этому сообщению нет. Эти числа — USER_REPORTED, не локальные измерения.

## Что найдено и изменено

| Вопрос | Подтверждено кодом | Изменение 0.5.3 |
|---|---|---|
| Регрессия относительно ~45 s | В старых загруженных логах 5-rank CPU/token/SDPA первый завершённый forward занимал 45.17/45.35 s. Safe exchange тогда был FP32. Новый normalized FP16 wire добавляет MAX all-reduce для V в каждом attention block (50 DiT blocks). Его вклад в 35 дополнительных секунд НЕ измерен. | Default sequence wire снова FP32; FP16 остаётся явным вариантом в Advanced. SDPA/FP16 Safe GEMM не переводятся этим в FP32 compute. Это возврат пути обмена, не полный rollback всех изменений. |
| Keep оставляет VRAM | GPU/ATS finish_sampling раньше сохранял live graph; idle retention работал только при CPUOffloadPolicy. | Между runs сохраняются CPU-only локальные DTensor shards, buffers, Safe GEMM/QK scalars. Live GPU/ATS graph удаляется; ATS pool принадлежит только ему. Восстановление из RAM возвращает первоначальный активный placement. |
| Prefetch «не действует» | Public FSDP prefetch lists выставлялись, но auto memory estimate мог уменьшить запрос 1/2 до 0. Не было простого UI-пояснения. | Requested/effective/reason видны в UI и rank-0 `powershard_prefetch` log. В Advanced есть auto/manual; manual не уменьшает запрос. Profiler показывает FSDP forward-prefetch ranges и NCCL events. |
| Задержка после вычислений | Были только общий forward и host IPC wall time. По ним нельзя выделить CUDA wait, результат или файловый I/O. | Добавлены раздельные wall phases, worker request, output serialization, host wait/read, progress I/O, RAM resume. MLP off/manual больше не опрашивает allocator в каждом блоке; progress JSON writes ограничены интервалом 0.5 s (first MLP/status/error пишутся сразу). |

Старый ~45 s — измеренный RPC в прежних логах, не автоматическое доказательство такого же времени sampler iteration в новом графе. CFG/conditioning могут давать несколько forward RPC на итерацию. Сравнивать нужно один checkpoint/precision, shapes, количество кадров, seed, CFG, steps и число реально выполненных RPC. Данные старых запусков: `reports/performance-0.5.1/uploaded-log-analysis.json`.

FP16 MAX не является ошибкой численной безопасности: он нужен, чтобы у ranks был одинаковый масштаб V перед half exchange. FP32 путь обходится без этого отдельного consensus. FP32 передаёт больше данных и может занять больше communication workspace; ни один вариант заранее не объявлен быстрее на вашем сервере.

## Keep: активный placement и хранение между runs

| Выбранный режим | Во время sampling | Между задачами при keep=true |
|---|---|---|
| CPU | CPUOffloadPolicy shards в RAM, активные группы на GPU | Те же CPU shards; очистка conditioning/Spectrum/CUDA cache |
| GPU | Локальные FSDP weights в VRAM | Только локальные CPU copies; live FSDP graph удалён |
| ATS | Scoped managed shards с CPU preference; active groups в обычной VRAM | CPU-only local cache; прежний managed pool не удерживается worker/factory |

Не выполняется полный model all-gather для сохранения. Нет полной 50 GB модели на каждом rank. Cache копирует только `parameter.to_local()` после проверки resharded state. Имена фиксируются до Spectrum gates, поэтому gate namespace не ломает восстановление. На resume нет чтения checkpoint weights, norm vectors, нового row-bound scan или bounds all-reduce; архитектурный header и INT8 quant metadata могут читаться. Shape/dtype/names проверяются, mismatch останавливает загрузку.

Parking/H2D/reconstruction выполняются **один раз между runs**, а не после каждой итерации. Они имеют отдельную задержку. CPU weights не переживают перезапуск ComfyUI, смену конфигурации/provider/checkpoint, повреждение CUDA/NCCL context или release с отключённым preserve. H3+Qwen RAM caches могут вместе заполнить 128 GB; базовые workflows оставляют Qwen `release`.

GPU context/NCCL, а при CPU policy и небольшие buffers, могут остаться в VRAM. «Ноль по nvidia-smi» не обещается. Host patcher учитывает измеренный idle allocated=0 как ноль, а не заменяет его старым активным VRAM budget. История rank RPC сохраняется на idle, без закрытия RAM cache.

## Почему token/Ulysses и ATS могут показывать похожую память

Token/Ulysses меняют attention communication, не размер локальных FSDP weights. Residual/MLP и CUDA allocator cache тоже могут определять общий peak сильнее, чем разница KV gather/all-to-all. На пяти ranks 56 heads дополняются до 60 для Ulysses; это корректный padding, не дублирование полной модели. Одинаковый sampled nvidia-smi usage сам по себе не доказывает неработающий Ulysses.

ATS не равен «все физические страницы всегда в RAM». Managed allocations и активные группы могут быть резидентны в VRAM. Torch allocated — логический объём, не физическая резидентность страниц. Значение 80–90% и 84 s само по себе не доказывает ошибку ATS или ускорение. Официальные [CUDA notes PyTorch](https://docs.pytorch.org/docs/2.12/notes/cuda.html) описывают page-fault/migration costs UVM и преимущество explicit placement, когда рабочая память помещается. Для этого сервера вывод пока требует trace и NVML/driver samples.

## Prefetch: что именно проверять

Согласно [FSDP2 PyTorch 2.12](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html), explicit prefetch выдаёт следующий all-gather раньше, в pre-forward текущего модуля. Даже при prefetch=0 возможен implicit overlap благодаря CPU run-ahead. Поэтому 0→1 не обязано дать ускорение или заметный прирост sampled VRAM.

Auto — безопасная оценочная настройка, не OOM guarantee. Если requested=2/effective=0, эксперимент с «двумя prefetch blocks» фактически не состоялся. Manual гарантирует выставление списка, но не overlap/ускорение и может дать OOM. Profiler ranges подтверждают вызов prefetch helper; чтобы доказать перекрытие, нужно смотреть timeline actual all-gathers и GEMM, а не только число ranges.

## Минимальный серверный A/B

После замены единственной папки custom node перезапустите ComfyUI и обновите страницу. UI schema 7 добавляет wire/prefetch-policy в конец Advanced; old six-widget tuning и seven-widget keep layout мигрируют. Явно сохранённые FP16 и MLP auto/manual настройки не переписываются.

Первый прогон на прежнем графе: **5 GPU, CPU placement, SDPA, token, sequence_comm_dtype=fp32, MLP=off, prefetch=0, keep=true**. Не меняйте вход, длину видео, CFG, seed или steps. Сравните не только первый вызов, но и последующие 2–3 forwards. Для prefetch сравните 0/1 при manual на том же wire/placement; при OOM верните auto/0. Затем отдельно сравните FP32/FP16 wire и token/Ulysses.

Генерация A/B файлов сама не отправляет задания:

```bash
python scripts/benchmark_cases.py /ABS/current.api.json --axis wire --output-dir reports/local-wire
python scripts/benchmark_cases.py /ABS/current.api.json --axis prefetch --output-dir reports/local-prefetch
```

Сначала маленькая аппаратная проверка GPU/RAM cache; нужны реальные выбранные CUDA-visible IDs:

```bash
python scripts/probe_h3_cuda.py --comfy ../.. --gpus 0,1,2,3,4 --weight-placement gpu --sequence-mode token --attention-backend sdpa --ram-roundtrip --output reports/local-cache-gpu.json
```

Для CPU/ATS меняется только placement, ATS проверяет capabilities и не подменяется CPU offload. Probe создаёт только tiny временные checkpoints и не читает production weights. В локальном CPU runtime он **NOT_RUN**. Для настоящего checkpoint `accept_h3.py --ram-roundtrip` проверяет сохранение PID, восстановление RAM cache и совпадение результата; размеры conditioning синтетические, это не реальная генерация и не её benchmark.

Логи после sampling сохраняются в `ComfyUI/output/powershard`. Сводка читает их без запуска GPU:

```bash
python scripts/summarize_timing.py /ABS/ComfyUI/output/powershard
```

Сводка берёт maximum across ranks для каждого RPC и не смешивает config/shapes/profiler. `None` означает отсутствие замера. Новые session/run reports с одинаковыми session_id/sequence не считаются дважды. Median включает собранные calls; cold/warm автоматически не угадывается.

Для одного диагностического forward запустите ComfyUI с `POWERSHARD_PROFILE=1 POWERSHARD_PROFILE_FORWARDS=1`. Trace добавляет overhead, его нельзя считать обычным замером скорости. Нужны API workflow, rank JSON/logs и trace хотя бы rank 0; все ranks лучше для обнаружения перекоса.

Если велик `finish_cuda_wait_s`, это ожидание ещё не завершившейся GPU/collective работы, а не доказательство «лишней синхронизации»: synchronize просто делает ожидание видимым. Большие `output_serialization_s`, host `output_deserialization_s`, `progress_write_wall_s` или `worker_request_s` вне forward требуют отдельного CPU/I/O анализа. Ranges NCCL и GEMM покажут, действительно ли карты простаивают между блоками. Потребление в ваттах не подтверждает kernel selection.

## Проверки этой ревизии

298 PASS, 16 SKIP: x86_64, Python 3.12, torch 2.12.0+cpu, native ComfyUI f1072eb0350638a3390ddb6afbcaa8c6b237c6fd. Проверены local-row cache ownership/bytes для пяти simulated ranks, strict restore validation, graph cycle collection, retention/cancel, CPU native H3 exact round-trip (включая canonical names под gates), H3/Qwen QK metadata restoration без weight reads, allocator-free off/manual MLP, UI migration, timeline-summary extraction и native sampler/clone contracts. Отдельный CPU contract: PASS. Undefined-name checks, compileall, JS syntax и patch whitespace: PASS.

16 SKIP относятся к Gloo socket transport, который среда не разрешает. CUDA/NCCL/POWER9/V100, реальный ATS allocator/residency, крупные checkpoints, качество production outputs и ускорение 80→45 s: **NOT_RUN / NOT_MEASURED**. CPU mocks не называются аппаратной приёмкой. История, source SHA и XML: `reports/performance-0.5.3/`. Incremental patch относительно 0.5.2: `review/changes_0_5_3.patch`.
