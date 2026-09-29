# Ограничения 0.5.0rc1

- RAM-профиль реализован и CPU-тестирован, **CUDA/nested FSDP2 + CPUOffloadPolicy на AC922 ещё NOT_RUN**. Ожидаемую экономию нельзя выдавать за замер.
- `workspace_mib` ограничивает оценку временного вычисления, не total VRAM. Live residual/output, full K/V, native Qwen masks, FSDP copy/all-gather, большие embedding и CUDA/NCCL способны вызвать OOM.
- Vision Qwen остаётся compute-replicated; token sequence удерживает глобальные K/V. INT8 embedding превращается в локальные floating shards; Linear остаются INT8.
- Модельный ATS allocator не добавлен. `weight_placement=ats` — обычный CPU offload и диагностическая операция. Новый direct-memory C++ probe не проверен без CUDA/nvcc.
- RAM-профиль может быть значительно медленнее: больше H2D, небольших GEMM, FSDP-групп; подготовка MLP-весов отключена. Spectrum/Sage изменяют математику независимо от профиля.
- Полное независимое выделение activations/workspace по категориям в VRAM не измерено; JSON разделяет известные параметры/cache и overlapping allocator snapshots, не складывает их.
- Retained sessions накапливают CPU history метрик; для длительного сервиса оставьте release_after_sampling=True либо периодически Release. Эта история не является копиями весов.
- В реальном browser ComfyUI новая панель ещё NOT_RUN. Контракты полей, migration и Apply проверены в JS DOM harness.

[Режимы, ограничения и приёмка](docs/RAM_ATS_UI_RU.md).

---

# Известные ограничения 0.4.0

1. **Аппаратная приёмка нового кода не выполнена.** Полная pretrained H3 generation, Qwen32B, FSDP/CPUOffloadPolicy/sequence/Spectrum на CUDA, NCCL recovery, реальные VRAM/PSS/throughput и POWER9 NVLink — NOT_RUN / NOT_RUN_ON_AC922. 195 CPU PASS это не заменяют. Ваши AC922 проверки старой серверной версии отдельно USER_REPORTED.
2. Доступен репозиторий 0.3 на `f4a53c8`, не неизвестная серверная копия. В нём нет LoRA descriptors. LoRA/ControlNet/произвольные patches не добавлены; совместимость серверных extensions не установлена. Не накладывайте полную папку поверх своих изменений без сравнения delta.
3. Qwen sequence math/RoPE/GQA проверены с подставленными collective tensors на CPU; реальный FSDP+sequence call order/offload ещё требует CUDA smoke. Vision compute реплицирован. Global hidden gather каждого слоя может дорого стоить; не обещается acceleration.
4. Qwen custom Flash/vllm GQA и FP32 vision сейчас идут через SDPA/math correctness fallback. Не следует ожидать использования custom FA всей моделью по одному выбранному имени. allow_fallback=false честно откажет этому контракту. Native vision processor сохраняет upstream CPU metadata conversions; их ускорение не заявлено.
5. Memory policy использует оценки. Не гарантирует отсутствие OOM и не сводит все параметры/activations к checkpoint/N. Длинное attention и FP32 residual/KV могут исчерпать память каждой GPU даже при CPU weights. Для мультимодального Qwen expansion до processor неизвестно. Generic activation-offload не реализован; отдельно реализован bounded CPU Spectrum history.
6. Spectrum — приблизительный экспериментальный адаптер, не полный порт upstream solver bridges. Пока только deterministic Euler; другим samplers оставлен ACTUAL FSDP. Default audio spectral blend0 всё ещё означает линейное прогнозирование audio hidden, не гарантию отсутствия искажений. Декодированные PNG/audio, fidelity и скорость на реальных weights NOT_RUN. На 4 шагах default policy не пропускает ни одного блока.
7. Spectrum local history budget действует per rank; абсолютный суммарный CPU лимит равен сумме заданных per-rank лимитов, а не одному history_mib на все процессы. Ни один rank не хранит полную target history специально. NCCL/allocator context idle encoder остаётся в VRAM. CPU history transfer корректный синхронный, overlap не заявлен.
8. BF16→FP16 веса проверяются на finite/range; непредставимые сами веса не исправляются clamp. FP16 rounding/underflow и ConvRot quantization требуют отдельной полноразмерной проверки качества. BF16→INT8 конвертера нет; принимается существующий INT8 checkpoint.
9. VAE не шардированы. Root/small buffers и активные groups/prefetch/dequant — дополнительная память. Реальный CUDA Tensor Core dispatch, а не только half operands, требует аппаратного profiler.
10. Runtime report checkpoint identity использует path/size/mtime/header hash. Полный hash многогигабайтных weights не считается каждый RPC; manual replacement с сохранёнными timestamps требует явной новой identity. Замена импортированных .so требует перезапуска ComfyUI.
11. Браузерное открытие новых UI workflows и аппаратная OOM/kill/timeout recovery не проверены. Сами JSON/wiring/old widget positions и CPU process recovery проверены. --cache-none заставляет выполнять graph, но может делать model loads холодными; не путайте cache hit с warm compute.
12. Gloo transport ранее был BLOCKED: Operation not permitted до FSDP. Ограничение не обходилось, отсутствие CPU distributed PASS не означает отсутствие CUDA поддержки.

## Исторические ограничения 0.3

Ниже сохранён старый отчёт. Его пункт 9 («text encoder не шардируется») относится только к прежнему loader, а не к новой реализации Qwen 0.4. Актуальные статусы перечислены выше.

0. Доработана доступная сохранённая база 0.2.0, не неизвестная серверная копия. В ней LoRA descriptors отсутствуют и LoRA явно отклоняется. Если на сервере они уже реализованы, требуется предоставленный diff и объединение; не заменяйте пользовательскую копию вслепую. Все новые физические GPU/provider tests NOT_RUN. CPU/native suite: 147 PASS, не доказательство FSDP/CUDA.

Новые ограничения attention: custom provider CUDA probe пока покрывает FP16, текущий head_dim, equal heads, noncausal cross/square causal; advanced options/GQA имеют correctness fallback. `return_attn_probs` использует math и может потребовать большой dense tensor по явному запросу. Возможное ускорение/качество Sage не измерено. UI picker в браузере NOT_RUN. Замена импортированного `.so` требует restart ComfyUI. См. [полный контракт](docs/MULTIGPU_ATTENTION.md).

1. **Полная генерация реального MiniMax H3 на трёх V100 не получена на этом стенде.** Нет GPU, AC922, pretrained checkpoints/энкодеров/VAE. Статус всех аппаратных режимов NOT_RUN, POWER9 — NOT_RUN_ON_AC922. Это не заключение о несовместимости.
2. CPU/Gloo init_process_group в Python3.11/torch2.12 попытался создать transport socket и получил Operation not permitted. FSDP forward не выполнялся. Ограничение не обходилось.
3. FP16 Safe исправляет воспроизведённый condition overflow и проходит tiny H3 numerical/native sampler tests. Устойчивость/качество всех реальных diffusion steps не доказаны. Scaling оставляет FP16 rounding/underflow; веса, сами не представимые в FP16, по-прежнему отклоняются при загрузке. Нельзя обещать эквивалентность BF16 checkpoint целиком.
4. CUDA/FSDP frozen I8, CPUOffloadPolicy и prefetch реализованы, но ещё не прошли hardware smoke. Probes проверяют их до H3. Полная dense распаковка в качестве fallback отсутствует.
5. FSDP-only повторяет compute на ranks. SP распределяет tokens в коде, но speedup не измерен. Global FP32 KV и Python chunk loops могут свести выгоду на нет.
6. С текущими FP32 residual/attention operands пик больше старой FP16 оценки. Dense scaled Linear временно создаёт FP32 текущую матрицу; INT8 ограничен rows. Статический размер checkpoint/3 не гарантирует размещения даже на 32 GiB V100.
7. CPU offload хранит только шарды по коду. Реальный суммарный PSS, pinned RAM, page-cache и staging peak надо измерить. Сумма RSS завышает физическую память shared mappings. Наличие NVLink на AC922 не доказывает эффективный маршрут каждого transfer.
8. NUMA auto задаёт CPU affinity при известной locality, но не жёсткий memory bind. bind использует numactl при доступном узле; иначе предупреждает и запускает с inherited policy. Неизвестная locality не угадывается. Выигрыш ни одного режима здесь не измерен.
9. Text encoder не шардируется. CPU FP32 H3 Qwen требует около 96 GiB только под веса плюс RAM загрузки/активаций; native_offload — отдельный непроверенный на V100 путь. VAE также не шардируются.
10. Не реализованы LoRA, ControlNet, произвольные сторонние patches/wrappers, tied checkpoint mapping, training, FP8/NVFP4/W4A8. Они явно отклоняются. Patcher поддерживает PowerShard MODEL, не любой MODEL.
11. ComfyUI SHA/version gate отсутствует. Изменившиеся реальные API могут потребовать адаптации; понятная capability error допустима, отказ только по номеру версии — нет.
12. CUDA OOM/kill-worker/cancel/recovery и отсутствие orphan на целевом сервере ещё не проверены. CPU subprocess тест доказал восстановление после ошибки в unpatched session и чистоту cloned policy, не устойчивость NCCL при отказах.
13. Benchmark hooks/traces существуют; реальные load/encoder/VAE/collective/transfer timings и rank VRAM отсутствуют. Перекрывающиеся profiler события не суммируются с wall time. Trace сохраняется лишь для первых двух RPC, с profiler overhead.
14. ppc64le wheels/source builds реально не запускались. Проверены CPython3.11/glibc tag metadata и подготовлены source routes. Torchaudio2.12 wheel отсутствует, но upstream torchaudio2.11 совместим с torch>=2.11; CPU import/resample и native sampler здесь проверены с ним.
15. Графы на 5 кадров — короткий shape smoke. Качество длинных роликов, реальные UI execution/cache и SaveVideo NOT_RUN. Автоматических загрузок всех моделей нет.
16. Сохраняются размеры/mtime/header hash локального checkpoint, но не пересчитывается 20–40 GB SHA256 при каждом вызове loader. Downloader проверяет полный hash; свои локальные файлы проверяйте отдельно.
