> Историческая документация до 0.5.0. Старые UI/CLI аргументы и результаты не относятся к текущей ревизии. Актуальные команды: [README_RU.md](README_RU.md), результаты: [AUDIT_REVIEW_2026-10-04.md](AUDIT_REVIEW_2026-10-04.md).

# Фактические проверки 0.4.0

Исходная база `f4a53c8`; Linux x86_64, Python 3.11.16, torch 2.12.0+cpu; ComfyUI `7a0b5eede3f9721c8faab290689893f36edc6d66`. См. `reports/environment-0.4.json`. Нет CUDA GPU/nvcc, AC922, пользовательского wheel или больших весов. Ни одной зависимости целевого сервера не меняли.

| Проверка | Результат |
|---|---|
| Полный CPU/native suite | **195 PASS, 0 FAIL/skip**, `reports/tests-0.4-native.xml` |
| Прежний standalone native sampler contract | PASS, `reports/legacy-contract-0.4.json` |
| MLP manual/auto/off, dense/INT8, большой residual | PASS; сравнение с unchunked той же representation |
| MLP 17 tokens / chunk2 / 10 INT8 row tiles | 10 деквантизаций при bounded preparation вместо 90; структурный счётчик, не GPU speedup |
| Qwen text/image/2-frame-video native classes | CPU PASS dense+INT8, FP32-reference тех же dequant weights |
| Полный Qwen BF16 header ↔ native meta model | PASS 902 keys/shapes, 50 decoder layers; без загрузки weights |
| Qwen native CLIP scheduled encode/clone/cache | CPU PASS, native model-management path |
| Qwen sequence 1/2/3/6, GQA, causal/text/multimodal RoPE | CPU PASS с подставленными collective payloads; **не NCCL/FSDP** |
| Spectrum gate skip/native heads/repeated run | CPU PASS, настоящий SamplerCustomAdvanced + отдельный subprocess |
| Spectrum local target ranges 1/2/3/4/6/9 | CPU PASS; reconstruction, не физические ranks |
| Spectrum few-step Euler4 | 0 forecasts (warmup3 + tail1), **не ускорение** |
| Provider fingerprint warm median, 5 замеров после cold | **199.03 → 0.765 ms** на CPU; `reports/provider-stamp-0.4.json` |
| Архив видео | PASS metadata extraction: 7 файлов, 5 уникальных; settings/latency UNKNOWN |
| AC922 3GPU FSDP и sequence, FA/vllm-fa/SDPA | **USER_REPORTED PASS** пользователем для его серверной версии; не приёмка 0.4 |
| Новый Qwen FSDP/offload/sequence на CUDA | NOT_RUN |
| Spectrum FSDP/sequence, качество audio/video | NOT_RUN |
| Custom vllm wheel POWER9/V100 | NOT_RUN_ON_AC922 |
| Реальная H3 generation/15 секунд/PNG comparison | NOT_RUN |
| VRAM каждого rank, PSS CPU shards, NVLink/NUMA speed | NOT_RUN_ON_AC922 / NOT_RUN_CUDA |

Текущий финальный suite не содержит FAIL. Промежуточная устаревшая проверка «offload только если это слово в имени workflow» исправлена на проверку wiring/policy; новый Spectrum consumer должен идти через patcher. Ранее Gloo transport получил Operation not permitted до FSDP и не переиспытан обходным транспортом: BLOCKED, не PASS.

Допуски новых numerical tests: Qwen FP16 vs FP32 при одинаковых dequant weights `rtol=.025, atol=.006`; sequence slices `.01/.002`; MLP high-magnitude `.003/16` (absolute=16 соответствует outputs больших порядков). Эти тесты проверяют вычисление, не оценку quantization quality относительно исходных BF16 весов. Native повтор одного seed одного CPU пути сравнивается точно; разные GPU kernels бит-в-бит не требуются.

Пользовательские 48s FSDP и 120–180s sequence дают **замедление sequence 2.5–3.75×**; общего manifest нет, объяснение конкретной разницы остаётся гипотезой до trace. MP4 1.625/5.166667s не содержат generation time. Новые counters/trace позволят разделить MLP GEMM/dequant, FSDP materializations, K/V packing, load, encode, VAE и total; estimated bytes не называются измеренным NCCL traffic.

Команды A/B, cold/warm/cache и матрица workflows: [PERFORMANCE_0_4.md](docs/PERFORMANCE_0_4.md). Source compute сокращён в коде, ускорение GPU этим не доказано. Ниже сохранены исторические результаты, без переноса их статуса на новые режимы.

## Исторические проверки 0.3.0

2026-09-16: Linux x86_64 CPU, Python 3.11.16 / torch 2.12.0+cpu, glibc 2.39. ComfyUI `7a0b5eede3f9721c8faab290689893f36edc6d66`. Отдельное test venv, никаких изменений серверного стека.

| Проверка 0.3.0 | Фактический статус |
|---|---|
| Полный CPU suite с актуальным native ComfyUI | 147 PASS; `reports/tests-0.3-native.xml`; 0 FAIL/skip |
| scale/layout/GQA/MQA/batch/varlen/mask/window/ALiBi/returns/dropout contract | PASS CPU, mock custom provider; НЕ CUDA kernel PASS |
| Произвольные 1/2/3/4/6/9/12 устройств, порядок/дубликаты/invalid/all | PASS Python inventory/runtime contracts; НЕ физические GPU |
| Native H3 SDPA/math и FP16-safe QK/V compensation | PASS CPU, реальные H3 instances и call counters |
| Native sampler + CPU subprocess, повторные seeds/conditioning, clone/disable | PASS, сохранились старые регрессии |
| CPU/Gloo world=1 | BLOCKED: transport socket Operation not permitted до FSDP forward; запрет не обходился |
| CUDA SDPA/Flash/custom vLLM/Sage/math | NOT_RUN: GPU отсутствуют; optional providers не установлены |
| FSDP 1..N, CPU offload, VRAM/PSS, NCCL recovery | NOT_RUN |
| AC922 POWER9/V100, wheel `2.7.2.post1+cu124` | NOT_RUN_ON_AC922: стенд/wheel недоступны |
| Реальная pretrained H3 generation / PNG quality / speedup | NOT_RUN: нет GPU/весов/энкодеров/VAE |
| GPU picker в живом браузере | NOT_RUN; static widget order/graphs проверены |

Допуски: FP32 contract reference `atol=2e-6, rtol=2e-5`; half mock-provider `0.003`; isolated CUDA probe неквантованных providers `0.007`, Sage `0.08` (это только короткий kernel smoke, не качество H3); native H3 high-magnitude SDPA test относительный `0.01`. Проверяются формы/dtype/finite; bitwise равенство различных kernels не требуется.

Скрипт `benchmark_attention.py` измеряет warm-up, CUDA events kernel+adapter, отдельный packing overhead, wall time, CUDA allocated/reserved/peak и CPU memory каждого выбранного устройства. Чистое kernel-only время требует profiler attribution и не выдумывается из общего wall time. H3 stage benchmark сохранён. Ни одного нового GPU latency/VRAM числа не получено.

Команды и политика сравнения PNG: [MULTIGPU_ATTENTION.md](docs/MULTIGPU_ATTENTION.md). `reports/acceptance-0.3.json` содержит машинные статусы, `environment-0.3.json` — окружение. Ниже **исторические результаты 0.2.0**; пример оценки для трёх GPU не ограничивает N и не является новой мерой VRAM.

## Исторические проверки 0.2.0

Дата: 2026-09-15. Linux x86_64, AMD EPYC 9V74 (9 видимых CPU), glibc 2.39, **Python 3.11.16**, torch 2.12.0+cpu, torchvision 0.27.0+cpu, torchaudio 2.11.0+cpu, Comfy Kitchen 0.2.34. Всё установлено только в отдельное test venv после dry-run. CUDA driver/nvcc/GPU отсутствуют. Исходный пользовательский стек не заменялся.

ComfyUI checkout: 36da3ff763687eab86a35e1019995dd1fb369b0d, без core изменений. Python 3.12 отчёты прошлого этапа сохранены как исторические; не подменяют результаты Python3.11.

| Проверка | Результат |
|---|---|
| Unit/native suite | **46 PASS**, reports/local-py311-tests.xml |
| Старый Linear condition overflow | Воспроизведён FloatingPointError; это ожидаемый regression test, не оставшийся FAIL |
| Новый condition path | FP32 finite; input=100000, W=1, output=2400000 |
| Residual >65504; attention/out_proj; MLP fc2; nonzero bias | PASS CPU numerical tests; heavy operations с half operands |
| Full tiny native H3 FP32-reference comparison | PASS text/masks/FL2VA/Ref2VA/audio_scale; FP16 Safe atol/rtol=0.003 |
| ConvRot group256 | PASS vs Kitchen eager и tiny H3 safe AV forward; I8 storage сохранён |
| ModelPatcher node → subprocess → extra_conds/preprocess | PASS; native H3 в отдельном CPU процессе, input >half limit |
| Native SamplerCustomAdvanced + BasicGuider/Scheduler | PASS tiny CPU AV; настоящий sampler/callback/x0/patcher_extension |
| Повтор seed5/6/5, изменённые conditioning, worker reuse | PASS; одинаковые seed/input совпадают в этом тесте, PID не пересоздаётся между steps |
| Clone / disable patch / отсутствие global patch | PASS CPU; исходный MODEL неизменён, disabled session не наследует safe policy |
| Capability detection | PASS текущий ComfyUI; удаление требуемого метода вызывает API error, версия не gate |
| Полная meta-модель по настоящему header | PASS: 50 main +2 refiner blocks, hidden5376, heads56×128 |
| CPU/Gloo 3 ranks | BLOCKED: transport socket Operation not permitted до FSDP forward |
| Single-rank CUDA Safe; 3-rank FSDP FP16/INT8 | NOT_RUN |
| 3-rank FSDP CPU offload/pinned/pageable/prefetch | NOT_RUN |
| AC922 topology/NUMA/H2D/D2H/offload | NOT_RUN_ON_AC922 |
| Реальный checkpoint + ComfyUI encoder → sampler → обе VAE → файл | NOT_RUN |
| CUDA OOM/cancel/kill/recovery, real model switch | NOT_RUN |
| Дополнительный compute speedup | NOT_RUN; измерений нет |

Ожидаемые исключения непатченой модели проверяются pytest.raises. Оставшихся FAIL в финальном CPU suite нет. Отдельный Gloo probe — BLOCKED, не PASS и не диагностированный баг PowerShard.

## Запрошенное сравнение режимов

| Режим реальной H3 | Load / preprocess / sec-step / VAE / total | Rank VRAM / CPU RAM | Throughput |
|---|---|---|---|
| FP32 baseline | NOT_RUN | NOT_RUN | NOT_RUN |
| Single GPU FP16 Safe | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 GPU FSDP | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 GPU FSDP + CPU offload | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 GPU FSDP + SP | NOT_RUN | NOT_RUN | NOT_RUN |

Ни одного предполагаемого benchmark числа в таблице нет. Длительность pytest — не latency H3 генерации.

## Оценки, НЕ измеренная VRAM

Размеры файлов по HF metadata: BF16 Pruned 40 225 724 176 B, INT8 ConvRot 20 970 379 616 B (FL2VA и Ref2VA одинаковый размер, разные hashes).
Прежняя header-based нижняя оценка local parameter share — примерно 12.51 GiB dense / 6.54 GiB INT8. Это не пик и не обещание размещения. При offload эти canonical shards находятся на CPU.

Полный бюджет: shard padding + buffers + активные gathered groups + root + prefetch + communication copy buffers + FP32 residual/activations + attention workspace + transient FP32 dense weight/dequant + CUDA/NCCL/allocator + host pipeline.

Scores chunk: heads × query_chunk × key_chunk ×4 B. Safe SP global K/V: 2 × tokens ×56×128×4 B; при 100000 tokens только эти два FP32 tensors дают около 5.34 GiB, без padding/copies. Это иллюстрация формулы, не длина измеренного workflow.

Worker JSON содержит local_bytes/device/pinned/storage_bytes, CUDA allocated/reserved/peak, CPU RSS/PSS/HWM/NUMA pages, largest gathered group и bound активных+prefetch параметров. Activations/workspace отдельно не атрибутированы: явно помечены и входят в allocator peak. Для доказательства отсутствия трёх CPU копий сравнить сумму PSS и суммы local CPU shard bytes после освобождения loading buffers; отдельно наблюдать загрузочный peak.

## Как измерять

[scripts/benchmark_transfer.py](scripts/benchmark_transfer.py) измеряет H2D и D2H каждого GPU, pageable/pinned, с warmup и CUDA synchronize. Сопоставьте с topology/NUMA, не называйте GB/s доказательством маршрута NVLink само по себе.

POWERSHARD_PROFILE=1 сохраняет CUDA traces и collective event timings первых двух RPC; nested/overlapped events не складываются с wall time.
POWERSHARD_BENCHMARK_DIR включает native node stage measurement: load/encoding/denoising/оба VAE/save/cache. Worker времена вложены в sampler. Агрегация — scripts/benchmark_summary.py.
Команды всех режимов и критерии: [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md).
