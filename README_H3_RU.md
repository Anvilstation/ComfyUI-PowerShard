# ComfyUI-PowerShard 0.5.3

MiniMax H3 в ComfyUI с FSDP2 sequence sharding, выбором любого непустого CUDA-visible набора GPU и отдельными MLP-нодами. Основной профиль: IBM AC922, POWER9 ppc64le, 6×V100 16 GB, 128 GB RAM, Ubuntu 20.04.5, существующие custom PyTorch 2.12 / CUDA 12.4 / FA2.

`keep_in_memory` теперь сохраняет веса **в RAM между задачами для всех трёх placements**. Для GPU/ATS сохраняются только локальные шарды, FSDP graph удаляется; следующая задача восстанавливает его из RAM без чтения весов checkpoint. FP32 sequence wire возвращён по умолчанию как путь исторического SDPA baseline; normalized FP16 доступен явно. Добавлены prefetch policy, объяснение auto-снижения и wall timers. MLP off/manual больше не опрашивает allocator в каждом блоке. **CUDA/NCCL, освобождение физической VRAM, скорость и ATS на AC922 здесь не проверены. Возврат 45 секунд не заявляется.** Разбор и команды: [PERFORMANCE_FIX_0_5_3_RU.md](PERFORMANCE_FIX_0_5_3_RU.md). Исправления Qwen/FA/cancel из [0.5.2](FIXES_0_5_2_RU.md) сохранены.

## Установка обновления

Сохраните прежнюю папку custom node, распакуйте папку `ComfyUI-PowerShard` в `ComfyUI/custom_nodes`, затем перезапустите ComfyUI и обновите страницу браузера. Не держите две копии custom node одновременно внутри `custom_nodes`. Ваши модели и output находятся в ComfyUI и в архив не включены.

Используйте существующее окружение ComfyUI. Этот пакет не устанавливает и не заменяет torch, CUDA, NCCL или custom FA2. Обновление всего `requirements.txt` ComfyUI для установки этой ноды не требуется. Проверка окружения: `python scripts/diagnose.py`. При необходимости анализа зависимостей: `python scripts/dependency_plan.py` выполняет только dry-run с защитой установленного torch.

## Ноды

| Нода | Рабочие параметры |
|---|---|
| PowerShardConfig | `gpu_ids`, `weight_placement`, `precision`, `attention_backend`, `sequence_mode` |
| PowerShardConfigTuning, optional | reserve GiB, prefetch 0/1/2, NUMA, strict attention, host wrappers, pin memory; sequence wire FP32/FP16, prefetch auto/manual |
| PowerShardH3Loader | checkpoint, config, `keep_in_memory` (по умолчанию true) |
| PowerShardH3FP16Patcher | FP16 Safe и глубокая диагностика finite |
| PowerShardH3MLP | MODEL → MODEL; `off / auto / manual`, default off, размер chunk |
| PowerShardQwenMLP | CLIP → CLIP; `off / auto / manual`, default off, размер chunk |
| PowerShardH3QwenLoader | checkpoint, config, precision, idle policy, размер CPU conditioning cache |

Режим `fsdp2` убран из UI: используется `fsdp2_sequence`. `timeout_s` и `allow_unverified` удалены из UI и экспортируемого config. У RPC/генерации нет пользовательского таймаута; отмена ComfyUI и завершение worker продолжают обрабатываться. Технические пределы изолированных compatibility probes и остановки subprocess не являются временем генерации. NCCL API сохраняет внутренний watchdog на 365 дней.

`gpu_ids=all` выбирает все GPU, видимые **процессу ComfyUI**. `5,2,0` выбирает ровно такой порядок ranks. Индексы соответствуют CUDA_VISIBLE_DEVICES; UUID также разрешены.

## Удержание и отмена

`keep_in_memory=true` в H3 Loader сохраняет здоровые workers между задачами. После sampling выполняются end_run и idle: conditioning/Spectrum history и неактивный VRAM-кэш освобождаются. При `cpu` CPUOffloadPolicy продолжает владеть RAM-шардами; при `gpu/ats` локальные shards/buffers копируются в CPU cache, FSDP graph и его ATS pool удаляются. Следующая задача восстанавливает активный placement из этого RAM cache. Это перенос один раз между runs, не на каждом шаге. Qwen `keep` действует аналогично; `cpu_shards` пока требует CPU placement. Конфигурация Loader имеет приоритет над legacy `keep_workers`. При смене checkpoint/config/patch/provider перезагрузка ожидаема; одинаковый запрос переиспользует владельца весов.

Отмена retained RPC возвращается сразу, но уже запущенный forward безопасно заканчивается в фоне без следующих шагов. Следующая GPU-задача ждёт этого завершения; статус показывает draining. CUDA/NCCL/transport error закрывает повреждённые workers. Отмена во время первоначальной загрузки не сохраняет незавершённую модель. Зависший kernel/collective не даёт гарантии завершения drain; принудительная остановка ComfyUI освобождает workers и RAM.

`PowerShardRelease` перед VAE по умолчанию сохраняет удерживаемые H3/Qwen веса в RAM, включая активные GPU/ATS модели. Чтобы освободить и RAM, выключите соответствующий preserve-флаг. Private FSDP offload policy не меняется на лету: используется CPU-only cache локальных shards и реконструкция graph. CUDA/NCCL context (и маленькие buffers при CPUOffloadPolicy) могут занимать VRAM: ноль по nvidia-smi не обещается. История rank RPC сохраняется при idle, поэтому логи можно получить без закрытия RAM-владельца.

## Три режима памяти

| weight_placement | Постоянные веса | Вычисления и активный блок |
|---|---|---|
| `gpu` | Локальные FSDP shards в VRAM выбранных карт | GPU; FSDP собирает нужную группу весов |
| `cpu` | Локальные shards в RAM, pinned по умолчанию | GPU; CPUOffloadPolicy переносит активные группы |
| `ats` | CUDA Managed Memory shards, CPU preferred; аппаратный ATS обязателен | GPU; active FSDP groups и activations используют обычную VRAM |

В каждом режиме распределены shards одной модели: полная H3 не создаётся в каждом worker. `cpu` означает хранение весов в RAM и вычисления на GPU. ATS выделяет настоящую managed memory через отдельный scoped pool; это экспериментальный путь, проверяющий поддержку драйвера **до** тяжёлой загрузки. Обычная UVA не считается доказательством ATS.

Для ATS автоматически выбирается native CUDA allocator только в отдельных ATS workers, поскольку scoped MemPool несовместим с cudaMallocAsync/expandable_segments. Настройки allocator основного процесса ComfyUI сохраняются. Нужен доступный `c++`/`g++`; маленький allocator компилируется без nvcc и CUDA headers. Отсутствие ATS вызывает явную ошибку, без подмены CPU offload.

ATS расширяет возможность хранить **веса** за счёт RAM. Активный блок, attention и video activations всё ещё должны помещаться в VRAM. Значение torch allocated для managed memory является логическим размером, а не физической резидентностью страниц. Резерв и память MLP — оценки; они не дают гарантии отсутствия OOM.

Таблица выше описывает активный sampling; при keep=true между задачами во всех режимах веса находятся в RAM. ATS не обещает более низкий NVML usage или ускорение. Token/Ulysses меняют attention exchange, а не количество локальных FSDP весов, поэтому одинаковое потребление памяти само по себе нормально. `prefetch_policy=auto` может снизить 1/2 до 0; смотрите событие `powershard_prefetch`, requested/effective и причину в UI. Для явного A/B существует `manual`, но он может дать OOM.

## Ulysses и численная устойчивость

H3 `sequence_mode=ulysses` поддерживает 3, 4, 5, 6 и другие размеры GPU-набора: heads дополняются нулевыми до кратности world, token padding убирается до softmax, фиктивные heads удаляются после обратного обмена. Token-режим собирает K/V. Qwen text использует свой token sequence path, vision вычисляется реплицированно; переключатель Ulysses относится к H3.

FP16 Safe включён по умолчанию в H3 Loader. Linear GEMM масштабируется, condition/norm/SiLU/residual сохраняют FP32. Default sequence exchange снова FP32; attention/GEMM остаётся FP16 Safe. При явном `sequence_comm_dtype=fp16` QKV/output нормализуются **до** half exchange, с дополнительным global V MAX на каждом block; большой V никогда не сужается без нормализации. Итоговый residual gather остаётся FP32. Wire dtype доступен в Advanced и Python/API. Chunking MLP управляется независимо от FP16 Safe.

`auto` проверяет vllm_flash_attn, flash_attn, SDPA, math по порядку на каждой карте; это выбор совместимости, не скорости. Для H3 Ulysses проверяются полное число heads и `ceil(heads/world)`; ошибка импорта `flash_attn_2_cuda` означает, что запрошенный интерфейс недоступен данному Python. При разрешённом fallback flash_attn также проверяет vllm_flash_attn перед SDPA. Strict attention запрещает подмену провайдера. Кнопка «GPU / attention / память» показывает реальные вызовы по группам и память ranks. CUDA/OOM ошибки завершают session и не повторяют блок на повреждённом context. Triton/torch.compile непосредственно не вызываются этой нодой.

Когда токенов слишком мало для непустого torch.chunk на всех ranks, проход явно использует реплицированные вычисления с FSDP sharding. Набор GPU не сокращается; это отражено в `duplicated_compute` и warning. Для нормальных video sequences доступен полный sequence path.

## Workflows и миграция

`workflows/ac922_6gpu_{gpu,cpu,ats}_{token,ulysses}.ui.json` — шесть стартовых UI-графов; рядом API-версии. В них INT8 H3/Qwen, FP16 Safe, MLP off, H3 keep=true, NUMA auto, prefetch=0, 512×512, 25 кадров, 8 steps, Qwen release перед H3. Выберите существующие checkpoint/VAE файлы и свой prompt. Для длинных sequences полный MLP может дать OOM: auto/manual включается явно в отдельной ноде. GPU performance/качество этих графов здесь не измерены.

Старые UI workflows известных форматов мигрируют при загрузке страницы через ComfyUI endpoint. Старые MLP-настройки переносятся в отдельные ноды, reserve/offload/NUMA — в актуальные controls. Для файлов и API:

```bash
python scripts/migrate_workflow.py old.ui.json new.ui.json
python scripts/migrate_workflow.py old.api.json new.api.json
```

Неизвестный позиционный layout или старые связанные advanced widgets требуют named API export либо ручного обновления. Прямой POST старого API graph в `/prompt` обходится без frontend migration: предварительно используйте CLI. `scripts/run_api_workflow.py` мигрирует named API автоматически.

## Проверка на сервере

Команды выполняются в Python вашей ComfyUI, из этой папки; замените пути checkpoint на свои.

```bash
python scripts/probe_h3_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2 --sequence-mode ulysses --weight-placement cpu --output reports/local-3gpu-cpu-ulysses.json
python scripts/probe_h3_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2,3 --sequence-mode ulysses --weight-placement gpu --output reports/local-4gpu-gpu-ulysses.json
python scripts/probe_h3_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2,3,4 --sequence-mode ulysses --weight-placement cpu --output reports/local-5gpu-cpu-ulysses.json
python scripts/probe_h3_cuda.py --comfy /opt/ComfyUI --gpus all --sequence-mode ulysses --weight-placement ats --output reports/local-6gpu-ats-ulysses.json
python scripts/accept_h3.py --comfy /opt/ComfyUI --checkpoint /path/to/H3.safetensors --profile profiles/ac922-v100.json --gpus all --weight-placement cpu --sequence-mode ulysses --debug-finite --lifecycle --output reports/local-real-h3-cpu-ulysses
python tests/audit_memory.py /path/to/H3.safetensors --comfy /opt/ComfyUI --sets 3 4 5 6 --placement cpu --sequence-mode ulysses --forward
```

Tiny probe проверяет FP16/INT8 FSDP2, 7 heads, неравномерные токены, пять последовательных forwards, high-range conditioning, native audio/video output и сохранение managed pointers для ATS. Он не проверяет качество настоящего checkpoint. Повторите пары token/Ulysses и cpu/gpu/ats с одинаковыми inputs; `scripts/compare_runs.py` сравнивает latents только при совпадении seed, input hash и checkpoint header hash.

При повторном NaN включите `POWERSHARD_DEBUG_FINITE=1` и сохраните output/powershard rank logs. При OOM смотрите worker CUDA peak, driver memory по UUID, CPU RSS и largest active group. `tests/audit_memory.py` теперь действительно запускает workers до load-only замера и читает их статистику.

Команды исследования производительности и приоритеты доработок находятся в [PERFORMANCE_FIX_0_5_1.md](PERFORMANCE_FIX_0_5_1.md).
