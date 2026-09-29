# PowerShard 0.4.0 — результат доработки

Работа продолжена после временной недоступности среды. Существующее дерево 0.3.0 (`f4a53c8a2e17c4f01741e65ef46deffc36531071`) и история сохранены. Старые workflows не изменены. Новые функции реализованы в том же пакете; core ComfyUI и установленный стек целевого сервера не изменялись.

**Итог проверки: 195 PASS, 0 FAIL/skip на x86_64 CPU, Python3.11.16 / torch2.12.0+cpu. Полной pretrained генерации и новой аппаратной приёмки на AC922/V100 здесь нет.** Доступны точные команды для сервера и статусы NOT_RUN, а не вымышленные VRAM/latency.

## Что изменено

| Область | Файлы |
|---|---|
| MLP manual/auto/off, scoped INT8/dense preparation | powershard/operations.py, fp16_safe.py, patch_config.py, memory_policy.py |
| Бюджет памяти, FSDP metrics, объединение K/V | fsdp_backend.py, attention.py, telemetry.py |
| Qwen native CLIP/compute/FSDP/idle/cache | qwen.py, qwen_adapter.py, qwen_backend.py, conditioning_cache.py |
| Spectrum MODEL patcher, sampler boundary, worker gates | spectrum_config.py, spectrum_host.py, spectrum.py, comfy_adapter.py, host_guard.py |
| Общая session, роли, worker control/phase lock/fingerprint | runtime.py, worker.py, nodes.py, diagnostics.py, web_api.py |
| Отчёты, приватность inputs, benchmark plan/media audit | reporting.py, benchmark.py, scripts/accept_qwen.py, benchmark_cases.py, benchmark_host_overhead.py, extract_video_metadata.py, run_api_workflow.py |
| Workflows и regression tests | scripts/build_workflows.py, workflows/*qwen*, *long_auto*, *spectrum*, tests/test_mlp_policy.py, test_qwen_native.py, test_spectrum.py, обновлённые legacy tests |
| Документация/версии/provenance | README_RU.md, ARCHITECTURE.md, COMPATIBILITY.md, BENCHMARKS.md, KNOWN_LIMITATIONS.md, docs/PERFORMANCE_0_4.md, sources.lock.json, models/*qwen*, reports/*0.4* |

Точный список delta находится в `reports/changes-0.4.txt` и Git patch архива. В архиве также source tree и полный Git bundle для сравнения с вашей копией. Если серверная версия содержит дополнительные изменения, сначала проверьте patch и объедините изменения; неизвестный серверный diff здесь не был доступен.

## Ответы по существу

1. В приведённом сравнении fsdp2sequence **медленнее**, а не быстрее: 120–180s против 48s. Нет идентичного benchmark manifest, поэтому нельзя приписать всю разницу одному kernel. По коду sequence добавляет global K/V на каждом блоке, packing/padding и уменьшает локальные GEMM. FSDP-only повторяет compute, но избегает этих дополнительных sequence exchanges. Число занятых GPU не доказывает speedup.
2. MLP chunks экономят workspace за счёт дополнительных вызовов. Добавлен понятный **off**, оставляющий Safe включённым, а также auto. Увеличение chunk могло ускорить ваш случай; теперь bounded preparation устраняет повторную деквантизацию одних весов между chunks. Unit-счётчик: 10 dequant tiles вместо 90 на данном test shape. Фактическая скорость V100 NOT_RUN.
3. Добавлен **PowerShard H3 Qwen Loader** с реальным CLIP и native text/image/video conditioning. FP16 и INT8 ConvRot, meta/local-row FSDP, CPUOffloadPolicy, idle CPU shards, cache и отдельный sequence path реализованы. На реальной полной header meta-модели совпали все 902 keys/shapes; numeric/sampler/CLIP проверки используют маленькие настоящие ComfyUI классы. Pretrained 32B inference NOT_RUN.
4. CPU offload переносит шарды весов, но оставляет текущие activations/KV/workspace на GPU. 48 ГБ суммарно могут закончиться, и каждой карте нужно отдельно уложиться в её лимит. Auto memory policy вычитает reserve/активации/communication до prefetch/MLP workspace; нельзя гарантировать отсутствие OOM. Полноценного generic activation-offload без backward не заявлено.
5. Добавлен **PowerShard Spectrum → MODEL**, по умолчанию disabled. Config применяется в каждом worker, решение ACTUAL/FORECAST согласовано до FSDP. Gates стоят снаружи FSDP child: skip реально пропускает DiT __call__/all-gather. History только local target audio/video rows; native final heads всегда выполняются. Для Euler7 CPU integration получены реальные forecasts; Euler4 default policy даёт 0. На CUDA/реальном качестве это NOT_RUN.
6. Варианты ускорения теперь можно сравнивать по одной оси: MLP, attention, offload, prefetch, sequence, затем Spectrum. Cache Qwen уменьшает повторный encoding, а не denoising. Скрипты отделяют loading/encoding/denoising/VAE/cache. Нет обещания 3×, «ровно трети VRAM» или бесплатного ускорения CPU-history.

## FP16, INT8 и FSDP

Исправление прежнего condition_proj overflow сохранено: input/residual и condition_proj FP32, нормализация/gating FP32; тяжёлые attention/MLP используют scaled FP16 GEMM с FP32 восстановлением и finite check на границе RPC. INT8 parameters/scales остаются шардированными, metadata не теряется. Новая подготовка half tiles ограничена активным MLP и очищается в finally, в том числе при ошибке. Это не постоянная dense модель.

FSDP2: world_size из выбранных UUID, один worker/GPU, mesh(N,), блоки/aux/root reshard_after_forward=True, no_grad, явный reshard/assert. CPU offload — штатная policy над local shards с первого materialization. Prefetch 0/1/2; в auto effective depth согласован ranks. H3 и Qwen share runtime, но имеют отдельные native compute adapters. На одной GPU нет меж-GPU шардирования. На N>1 код шардирует weights; **факт нового H3/Qwen шардирования на физических GPU здесь не измерен**.

Spectrum не принимает LoRA descriptors, потому что их нет в предоставленной базе 0.3. Они не удалены из работающей реализации: серверный код с ними не предоставлен. Arbitrary wrappers/control/hooks всё ещё отклоняются до collectives. Полной совместимости со всеми сторонними нодами не заявлено.

## Фактическое оборудование и результаты

- Наш стенд: x86_64 CPU, Python 3.11.16, torch 2.12.0+cpu, ComfyUI `7a0b5eede3f9721c8faab290689893f36edc6d66`; CUDA/NCCL недоступны. Отдельное test env, серверный стек не затронут.
- PASS: 195 CPU tests; standalone native sampler contract; media metadata; bounded cache, clone/disable/run reset, native image/video Qwen, RoPE/GQA/causal partition, Spectrum child skip и audio/video outputs, прежний overflow regression.
- FAIL в финальном suite: **0**. Предыдущие неверные предположения harness исправлены; тест не объявляется hardware PASS.
- BLOCKED: прежний Gloo socket Operation not permitted до FSDP; ограничения не обходились.
- NOT_RUN_ON_AC922: POWER9 topology/NUMA/H2D/D2H, пользовательский `vllm_flash_attn-2.7.2.post1+cu124-cp311-cp311-linux_ppc64le`, CPUOffloadPolicy/NVLink и реальные peaks.
- NOT_RUN: новая полная H3 generation, pretrained Qwen32B, CUDA/NCCL/FSDP/Spectrum/sequence quality, OOM/kill recovery на GPU, реальный VRAM rank0/1/2 и CPU PSS модели. Никаких значений этих метрик не выдумано.
- USER_REPORTED: ваши успешные AC922 3GPU fsdp2 и fsdp2sequence + vllm-fa/fa/sdpa. Это свидетельство вашей версии, не испытание нового delta.

Полезное дополнительное распараллеливание реализовано в sequence-коде; измеренного GPU speedup нет. Измерено лишь сокращение CPU provider_stamp warm median 199.03→0.765ms на этом стенде, и число операций в маленьких тестах. Таблицы/допуски: BENCHMARKS.md.

Из `video(3).zip`: 7 файлов/5 уникальных, 960×544, длительность контейнеров 1.625/5.166667s; workflow/seed/backend/time UNKNOWN. Розовые артефакты нельзя по этим данным приписать shim. Новые workflows сохраняют PNG до видеокодирования.

## Первый запуск

Сохраните свои текущие изменения вне ComfyUI/custom_nodes, сравните delta от f4a53c8; не перезаписывайте серверную копию вслепую. Зависимости не обновляйте автоматически. Используйте Python активной ComfyUI и уже установленный custom wheel.

Из каталога PowerShard:

```bash
python scripts/diagnose.py --comfy /ABS/ComfyUI --output reports/ac922-environment.json
python scripts/probe_devices.py --gpus 0,1,2 --timeout 180
python scripts/accept_qwen.py --comfy /ABS/ComfyUI --checkpoint /ABS/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors --gpus 0,1,2 --cpu-offload --idle-policy cpu_shards
bash scripts/launch_comfy.sh /ABS/venv/bin/python /ABS/ComfyUI
```

Откройте **workflows/fl2va_qwen_int8_offload.ui.json**. Выберите H3 FL2VA Pruned INT8 ConvRot, Qwen H3 INT8 ConvRot, оба native VAE и GPU. Сначала Spectrum не подключён; sample Euler20/seed44/25frames. gpu_ids=all выбирает все видимые карты, без нового максимума6. Следующим A/B используйте fsdp2_sequence при неизменных inputs. Spectrum включайте отдельным графом fl2va_spectrum_euler и сравнивайте PNG/audio/actual-forecast counts. Полный набор команд, включая --cache-none и phase profiler: [PERFORMANCE_0_4.md](docs/PERFORMANCE_0_4.md).

## Исторический отчёт 0.3.0

Ниже оставлен отчёт предыдущей версии; он не описывает новые Qwen/Spectrum функции и не заменяет свежие статусы выше.

2026-09-16. Изменён существующий проект. Аппаратная приёмка **не выполнена**: здесь нет CUDA GPU, POWER9, пользовательского wheel или pretrained H3 weights.

## Исходная база

В рабочем каталоге был только приложенный shim. Восстановлен сохранённый PowerShard 0.2.0 с Git bundle, HEAD `9694964e68d58f7aa5228817dc65ecff9e7cda76`, чистое дерево. История сохранена. Более поздняя серверная копия недоступна. В этой базе **LoRA descriptors отсутствуют**, прежний явный отказ LoRA сохранён. Совместимость неизвестных серверных доработок не проверена; нельзя перезаписывать их вслепую.

Архив shim прочитан, оригинал не изменён. В нём только Python/pyc, нет CUDA wheel/исходников ядра. Пользовательская сборка `vllm_flash_attn-2.7.2.post1+cu124-cp311-cp311-linux_ppc64le` не устанавливалась и не заменялась.

## Изменённые файлы

| Область | Файлы |
|---|---|
| CUDA-visible GPU selection / UUID / N ranks | powershard/devices.py, config.py, runtime.py, worker.py |
| FSDP / shards / preflight / memory estimate | fsdp_backend.py, preflight.py, validation.py, checkpoint.py, comfy_adapter.py |
| Общий attention contract, math/SDPA | attention_contract.py |
| Flash, Sage, пользовательский vLLM | attention_providers.py, vllm_adapter.py |
| Probe / общая policy / fallback / counters | attention_policy.py, attention_probe.py |
| Реальные H3 call sites / safe QK/V scaling | attention.py |
| Config / inventory API / GPU picker | nodes.py, web_api.py, web/powershard.js, корневой __init__.py |
| Диагностика / optional NUMA | diagnostics.py, topology.py |
| CLI / примеры / benchmark | scripts/probe_devices.py, probe_three.py, probe_h3_cuda.py, probe_cpu_fsdp.py, accept_h3.py, launch_comfy.sh, diagnose_attention.py, benchmark_attention.py, build_workflows.py |
| Регрессии | tests/test_attention_contract.py, test_attention_h3.py, test_devices.py, test_runtime_devices.py; обновлены test_portable.py и test_native_pipeline.py |
| Документы / результаты / версия | README_RU, ARCHITECTURE, COMPATIBILITY, BENCHMARKS, KNOWN_LIMITATIONS, docs/MULTIGPU_ATTENTION.md, reports/*0.3*, pyproject.toml, powershard/__init__.py |

Добавлены три API/UI workflow-пары: fl2va_all_sdpa, fl2va_all_vllm_offload, fl2va_subset_math. Старые JSON и class IDs сохранены. Новые widgets добавлены после старых.

Рабочие operations.py, FP16 patch policy, INT8 ConvRot representation и wire serialization не переписаны. ComfyUI core не менялся. Лицензии и авторство сохранены.

## Удалённые ограничения и оставшиеся ошибки

- Вместо ровно трёх устройств — любое непустое подмножество 1..N, включая all и переставленные ID. Максимума шесть нет.
- Numeric ID — индекс CUDA-visible GPU исходного ComfyUI, не физический индекс nvidia-smi. Mapping фиксируется UUID и логируется.
- N определяет workers, process group, DeviceMesh, shard bounds, collectives, ответы, метрики и shutdown.
- Отсутствие объявленного sm_70 — warning и фактический compute test, не самостоятельный запрет.
- Недостаточная прогнозная VRAM/reserve — warning; реальный OOM остаётся ошибкой.
- Неизвестная NUMA/locality/отсутствующий numactl — warning и inherited affinity.
- Удалён launcher assert Python==3.11. Packaging минимум 3.10 обусловлен синтаксисом кода; проверен Python3.11.
- Отсутствующий optional attention package не ломает импорт/отображение нод.

ComfyUI SHA gate уже отсутствовал и не возвращён. GPU/CPU name, architecture, версия CUDA/PyTorch и название сборки не являются whitelist.

Пустой/несуществующий GPU, неверные layout/shapes/GQA, повреждённые weights/metadata, non-finite result, отсутствующий необходимый API, несовместимый patch, нарушение output/protocol и фактический CUDA/NCCL failure по-прежнему дают конкретную ошибку. CUDA OOM/illegal access/device-side assert не маскируются повторным kernel вызовом: закрывается session, следующий запуск создаёт чистые workers.

## Исправления shim

Раздельные Q/K/V размеры, cumulative lengths и maxima; правильные batch boundaries/GQA; varlen-only custom entrypoint. Signature исследуется единожды. Scale/dropout/causal/window/ALiBi/deterministic/return requests либо передаются, либо получают семантически эквивалентный fallback. Unknown kwargs — ошибка.

Нет sys.path shadowing, sys.modules подмены и fake package version. Непрозрачный Flash S_dmask не называется probabilities: canonical output/LSE/probabilities выдаёт math. Rectangular causal alignment определён явно; boolean causal не копируется вслепую. Пример миграции shim с ручной резервной копией — в docs/MULTIGPU_ATTENTION.md.

Attention установлен **на instances реальной H3 внутри workers**, main DiT и token refiner. Requested/effective, provider origin/version/probes, причины и counters `dit:<provider>` / `token_refiner:<provider>` доступны в отчётах. Нулевой counter не означает Flash enabled.

Короткие CUDA/numerical probes выполняются отдельными процессами на каждом выбранном UUID до NCCL workers. Policy общая для всех ranks. AUTO не включает Sage и не обещает fastest. На недоступный provider при allow_fallback=true есть SDPA/math; false останавливает явно запрошенный путь.

Ограничение: production custom-provider probe пока покрывает FP16/head_dim текущего H3, equal heads, noncausal cross/square causal. Advanced options/GQA/dropout/window/ALiBi/rectangular causal покрыты CPU adapter tests, но ещё не сертифицированы на данном CUDA kernel; dispatcher выбирает точный SDPA/math с причиной. Это не молчаливое игнорирование параметров.

## FP16 Safe, INT8 и FSDP

FP32 condition/residual/norm/SiLU/native final islands сохранены. Linear/MLP — прежние scaled half GEMM. Перед fused attention Q/K делятся на степени двух по аналитической RMSNorm/RoPE границе из маленьких norm vectors checkpoint. `softmax_scale` компенсируется ровно один раз. V масштабируется GPU tensor и после attention восстанавливается в FP32. Нет per-block .item()/finite CPU sync, clamp или nan_to_num. Deferred finite tracker остаётся на RPC boundary.

INT8 storage/scales/ConvRot metadata не меняются; decoder не распаковывает генератор целиком. Shard по выходным строкам сохраняет column groups. CPUOffloadPolicy и prefetch 0/1/2 сохранены. N=1 использует те же FSDP hooks и NCCL infrastructure, но не называется меж-GPU шардированием. N>1 — один mesh (N,), block/root reshard_after_forward=True.

FSDP-only повторяет compute. Sequence backend использует все выбранные ranks; при пустом token rank — согласованный FSDP-only forward, не сокращение GPU count. Изменение Config/patch/provider обновляет session fingerprint и workers; clone не меняет другие ветки.

## Фактические результаты

Стенд: Linux x86_64 CPU, glibc2.39, Python3.11.16, torch2.12.0+cpu, torchvision0.27.0+cpu, torchaudio2.11.0+cpu, Kitchen0.2.34, Aimdo0.5.5. ComfyUI `7a0b5eede3f9721c8faab290689893f36edc6d66`. Зависимости установлены только в отдельное test venv после dry-run.

| Проверка | Результат |
|---|---|
| Полный CPU/native suite | **147 PASS**, 0 FAIL/skip; reports/tests-0.3-native.xml |
| Старые FP16/INT8/native AV H3, extra_conds, sampler, CPU subprocess, clone/repeated jobs | PASS; реальные маленькие H3 modules, не pretrained generation |
| Scale/layout/GQA/MQA/masks/varlen/returns/dropout | PASS CPU contracts; custom provider — mock, не CUDA |
| 1/2/3/4/6/9/12 устройств, order/invalid/all, N workers | PASS Python contracts; не физические GPU |
| Native H3 SDPA/math counters и safe scaling | PASS CPU |
| Python compile, JS syntax, Git whitespace, legacy widget order | PASS |
| CPU/Gloo world=1 | BLOCKED: socket Operation not permitted до FSDP forward; запрет не обходился |
| CUDA kernels / NCCL / FSDP N=1..N / CPU offload / OOM recovery | NOT_RUN: нет GPU |
| POWER9/V100 custom wheel / NVLink/NUMA transfer | NOT_RUN_ON_AC922 |
| Pretrained H3 generation / VAE PNG / Sage quality / rank VRAM / CPU offload PSS / speedup | NOT_RUN |
| GPU picker в живом браузере | NOT_RUN; static/syntax checks PASS |

Первоначальный недостающий comfy_aimdo установлен только в test venv; финальные тесты повторены. Номер версии не служил причиной пропуска. Ни одного придуманного GB/s, VRAM или sec/step числа нет. CPU suite не доказывает CUDA/FSDP.

## Первый запуск

Сначала сохраните серверную копию и сравните с доступной базой 0.2.0. Если появились LoRA descriptors/поздние fixes, нужен merge их diff, не полная замена. Пользовательский wheel оставьте установленным. Старый глобальный shim отключайте вручную после резервной копии, затем перезапустите ComfyUI.

Из ComfyUI/custom_nodes/ComfyUI-PowerShard, Python активного окружения:

```bash
python scripts/diagnose.py --comfy ../.. --output reports/server-environment.json
python scripts/diagnose_attention.py --gpus all --backend vllm_flash_attn --no-allow-fallback --output reports/server-vllm.json
python scripts/probe_devices.py --gpus all --attention-backend sdpa --timeout 180
bash scripts/launch_comfy.sh "$(command -v python)" "$(realpath ../..)"
```

Открыть `workflows/fl2va_all_sdpa.ui.json`, выбрать свои локальные H3 INT8 checkpoint, родной text encoder и оба VAE. Для custom wheel/offload — `fl2va_all_vllm_offload.ui.json`. all можно заменить допустимым набором, например 5,2,0. Новые графы сохраняют PNG из VAE до video encoder.

Полные команды FSDP/offload smoke, transfer/kernel benchmark и установки локального wheel без смены torch: [docs/MULTIGPU_ATTENTION.md](docs/MULTIGPU_ATTENTION.md).

**Итог:** изменения реализованы в доступном проекте; аппаратная приёмка, совместимость неизвестных серверных доработок и POWER9/V100 wheel ещё не подтверждены. Полной H3 генерации и измеренного ускорения здесь не получено.
