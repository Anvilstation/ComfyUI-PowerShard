# Отчёт: RAM, ATS и интерфейс PowerShard — 2026-09-29

**Реализован проверочный выпуск 0.5.0rc1 в существующем проекте. На AC922 не установлен. GPU-экономия и скорость пока не измерены.**

## Основа и сохранность

Отдельная ветка `work/ram-min-ui-20260929`, worktree `PowerShard-RAM`. Исходный репозиторий `ComfyUI-PowerShard` на `9da90b8cc4f5247bab6ce121f48088c90014702b` не изменён. Присланный вариант C и проверенные поправки предыдущего аудита включены в commit `bb75376`, от него выполнена новая работа. Снимки A/B/C/experimental и прежний аудит не перезаписаны. Core ComfyUI, torch/CUDA/NCCL, drivers, модели и пользовательский wheel не заменялись.

Составлен и исполнен промпт `docs/TASK_PROMPT_RAM_UI_RU.md`. Пользовательские API/class IDs и прежние widgets сохранены; новые поля append-only. Нет нового project/backend вместо FSDP.

## 1. Максимум весов в RAM

Добавлен `memory_profile=ram_min`: штатный CPUOffloadPolicy с локальными DTensor CPU-шардами; attention/MLP отдельными вложенными FSDP-группами, все группы включая root reshard=True. Explicit prefetch 0, prepared MLP weight cache выключен, MLP auto, dense Linear conversion по output rows. INT8 Linear сохраняет I8/scales/ConvRot, деквантизация ограничена активными строками.

Worker conditioning-кэш перенесён в ограниченный CPU LRU; исходные tensors независимы от mmap и in-place mutations. Устранён лишний clone непосредственно после новой межустройственной копии. GPU input/result освобождаются после RPC. Spectrum history в этом профиле хранится на CPU. В конце run выполняется phase-only idle/empty_cache, не синхронизация в каждом блоке.

SDPA исполняется по частям Q/heads, с полным K/V и сохранением causal positions, masks, scale, ALiBi/window и dropout. Math GQA больше не создаёт целиком repeat_interleave K/V на уровне Python. Custom FA/vLLM/Sage остаются явно выбранными providers; diagnostics показывают фактические `sdpa_tiled` вызовы. OOM/illegal memory access не скрываются fallback.

Планировщик учитывает свободные cache blocks native allocator, чтобы после первого прохода auto ошибочно не уменьшал chunk до 1 только из-за низкого driver-free. Это оценка, не admission check.

**Что остаётся на GPU:** активные веса/родительские группы, residual/output, полные KV в token sequence, часть native masks/positions/buffers, dequant/cast workspace, CUDA/NCCL. `workspace_mib` — не общий лимит VRAM и не гарантия отсутствия OOM. Максимум-в-RAM означает агрессивную policy, не доказанный теоретический минимум памяти всей H3. Локальные Qwen INT8 embedding-шарды по сохранённой реализации floating: это отдельно отмечено.

## 2. ATS

Изучены CUDA 12.4 Unified Memory и PyTorch 2.12 CPUOffloadPolicy/MemPool. ATS возможен как отдельный путь allocations и требует реального server-test kernels/FSDP/NCCL. Он не превращает любой CPU tensor в допустимый CUDA operand и не устраняет нехватку RAM/VRAM workspace.

Глобальный allocator не подменялся. **Модельный ATS backend не реализован.** Существующее `weight_placement=ats` явно подписано «CPU offload + диагностика». Добавлены standalone `malloc`/`cudaMallocManaged` CUDA integrity/cold/warm probes с capability check и отдельными процессами по GPU UUID. В среде результат NOT_RUN: нет CUDA GPU/nvcc. Это не PASS ATS и не измеренный NVLink.

## 3. Интерфейс

Русские названия нод, tooltip для каждого параметра, единая metadata schema, панель «Параметры и пояснения» с группами «Основное», «Память», «Дополнительно», «Совместимость». Понятные подписи provider/профилей и предупреждение об overrides RAM-профиля. Старые API enum tokens/сокеты/порядок widgets не изменены, миграция версии C сохранена.

Полноценный browser ComfyUI здесь NOT_RUN. Node.js DOM harness проверяет создание панели, Apply, подписи, отсутствие изменения widget IDs/сериализации и миграцию старых массивов.

## 4. Скорость и качество

Практическая таблица вариантов и компромиссов — `docs/RAM_ATS_UI_RU.md`. В первую очередь сравнивать RAM-профиль с custom offload, фактический FA/vLLM/SDPA, MLP chunk 4096/8192/16384, prefetch 0/1, subset GPU и NUMA/pinned. Нельзя обещать ускорение от RAM или большего N. Benchmark generator получил `--axis memory` и защиту от сравнения двух фактически одинаковых offload/MLP режимов под ram_min.

Качество: парные запуски с одинаковыми seed/inputs/weights/sampler/schedule, Spectrum выключен как baseline, PNG до видеоencoder и отдельное audio. Quantization/Sage/Spectrum — отдельные источники приближения. Не добавлены непроверенные Turbo/LoRA/ControlNet или универсальный совет увеличивать CFG/steps. Реальная оценка H3 видео здесь NOT_RUN.

## Фактически проверено

| Проверка | Результат |
|---|---|
| Основной Python набор | 241 PASS, 0 FAIL, 101.93 s |
| Регрессии аудита исходного C | 58 PASS, 0 FAIL, 35.64 s |
| Дополнительный benchmark CLI тест | 1 PASS, 0 FAIL, 6.80 s |
| Итого выполненных test cases | 300 PASS; две группы могут покрывать общие свойства |
| Native H3 маленькой размерности | preprocess_text и повторный полный video/audio forward; PASS CPU |
| Native Qwen маленькой размерности | dense/INT8, text/image/video, RAM/custom; PASS CPU |
| Bounded SDPA/GQA/causal/masks/scale/dropout, cache budget | PASS CPU |
| FSDP group selection/hooks | CPU contract PASS; не реальное CUDA FSDP |
| JS settings/migration и workflow arrays | PASS, DOM harness |
| Полная реальная H3 генерация | NOT_RUN |
| Новые FSDP/CPUOffload на V100/AC922 | NOT_RUN_ON_AC922 |
| ATS kernel, custom vllm wheel, NVLink bandwidth | NOT_RUN_ON_AC922 |
| Измеренная VRAM ranks, CPU PSS полного checkpoint | NOT_RUN |

Среда: Linux x86_64, Python 3.11.16, torch 2.12.0+cpu, glibc 2.39. Native ComfyUI commit `7a0b5eede3f9721c8faab290689893f36edc6d66` использован как фактически доступный API, не version gate. Исторические успешные AC922 запуски пользователя не переименованы в наши новые benchmarks.

Raw файлы этого каталога: `pytest.xml/.log`, `audit-regressions.xml/.log`, `benchmark-cli.xml/.log`, `environment.json`, `ats-probe.json`, `status.json`. Время pytest не является временем генерации.

## Первый тест на сервере

1. Сохранить установленный PowerShard вне custom_nodes; не подключать две версии class IDs одновременно. Использовать свой существующий Python 3.11 и wheel.
2. Запустить из новой папки плагина: `bash scripts/launch_comfy.sh /ABS/venv/bin/python /ABS/ComfyUI`.
3. Открыть `workflows/fl2va_ram_min_int8.ui.json`. Указать реальные локальные H3/Qwen/VAE checkpoint и выбранные GPU. Начать с короткого примера, FP16 Safe on, Spectrum off.
4. Проверить per-rank `memory_settings`, `stage_cache.residency=cpu`, `execution.units`, actual attention counts, sharded evidence и CUDA peak. Команды diagnose/NCCL/accept_h3/accept_qwen/ATS/transfer полностью приведены в `docs/RAM_ATS_UI_RU.md`.
5. Сравнить custom offload и ram_min на одинаковых условиях через `scripts/benchmark_cases.py --axis memory`. Числа GPU в примерах не ограничение; допустим `all`.

## Изменённые и добавленные файлы этого этапа

- `ARCHITECTURE.md`
- `BENCHMARKS.md`
- `COMPATIBILITY.md`
- `KNOWN_LIMITATIONS.md`
- `README_RU.md`
- `docs/RAM_ATS_UI_RU.md`
- `docs/TASK_PROMPT_RAM_UI_RU.md`
- `powershard/attention_contract.py`
- `powershard/attention_policy.py`
- `powershard/config.py`
- `powershard/fp16_safe.py`
- `powershard/fsdp_backend.py`
- `powershard/memory_policy.py`
- `powershard/nodes.py`
- `powershard/operations.py`
- `powershard/qwen.py`
- `powershard/qwen_backend.py`
- `powershard/ui_schema.py`
- `powershard/web_api.py`
- `powershard/wire.py`
- `powershard/worker.py`
- `pyproject.toml`
- `reports/ram-min-2026-09-29/ (статусы, raw logs/XML, диагностика, этот отчёт)`
- `scripts/accept_h3.py`
- `scripts/accept_qwen.py`
- `scripts/ats_memory_probe.cu`
- `scripts/benchmark_cases.py`
- `scripts/build_ram_workflows.py`
- `scripts/build_workflows.py`
- `scripts/probe_three.py`
- `scripts/run_ats_probe.py`
- `tests/test_qwen_native.py`
- `tests/test_ram_profile.py`
- `tests/ui_settings_contract.cjs`
- `web/powershard.js`
- `workflows/fl2va_ram_min_all_sequence.api.json`
- `workflows/fl2va_ram_min_all_sequence.ui.json`
- `workflows/fl2va_ram_min_int8.api.json`
- `workflows/fl2va_ram_min_int8.ui.json`
- `workflows/ref2va_ram_min_fp16.api.json`
- `workflows/ref2va_ram_min_fp16.ui.json`


Ранее внесённые C/audit исправления находятся в baseline `bb75376`; текущий delta не предназначен для слепого наложения на произвольно отличающийся сервер. Установочный архив содержит одну папку плагина, а не прежние четыре audit-снимка.
