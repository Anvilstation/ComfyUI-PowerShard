> Историческая документация 0.5.0rc1. Профиль `ram_min` и описанные здесь UI/API относятся к старой версии. Текущие настройки и миграция: [README_RU.md](../README_RU.md); изменения: [PERFORMANCE_FIX_0_5_3_RU.md](../PERFORMANCE_FIX_0_5_3_RU.md).

# RAM-профиль и интерфейс PowerShard 0.5.0rc1

Реализовано поверх присланных изменений и исправлений аудита, не вместо проекта. Код исходной рабочей копии сохранён. Проверочный выпуск: CPU-контракты проверены, новое вложенное FSDP/CPUOffload размещение на AC922 ещё **NOT_RUN_ON_AC922**. Ваши прежние успешные FSDP/sequence + FA/vLLM/SDPA запуски — отдельный `USER_REPORTED`, не проверка этих изменений.

## Первый запуск

Сохраните текущую папку PowerShard вне `custom_nodes`. Распакуйте новую `ComfyUI-PowerShard` в отдельную проверочную установку ComfyUI либо перенесите проверенные изменения Git. Не держите обе версии плагина в `custom_nodes`: class IDs намеренно одинаковые. Новые зависимости для этого обновления не нужны. Не переустанавливайте torch/CUDA/NCCL и свой wheel.

Из каталога PowerShard:

```bash
bash scripts/launch_comfy.sh /ABS/venv/bin/python /ABS/ComfyUI
```

Python должен быть из действующего окружения ComfyUI. Скрипт запускает localhost:8188 с `--disable-dynamic-vram --use-pytorch-cross-attention`; это не принуждает worker attention к SDPA. Core ComfyUI не изменяется.

Откройте `workflows/fl2va_ram_min_int8.ui.json`. Выберите свои локальные H3/Qwen/VAE файлы. Начальный пример: `0,1,2`, 768×768, 25 кадров, seed 44, Euler/simple, 20 steps. Это небольшой проверочный пример, не гарантированно оптимальный schedule качества. `all` и любое непустое видимое подмножество допустимы. Для другого checkpoint/sampler сохраните свои проверенные параметры.

В двух Config-нодах, H3 и Qwen, выберите:

- `memory_profile=ram_min` — «Минимум VRAM · веса в RAM»;
- `workspace_mib=256`, `stage_cache_mib=64` для первого теста;
- свой проверенный `attention_backend`: SDPA, FlashAttention или custom vLLM;
- GPU через существующую кнопку выбора, либо `gpu_ids`;
- Qwen `idle_policy=cpu_shards`, если RAM хватает; `release`, если Qwen нужно удалить после кодирования.

Панель «Параметры и пояснения» группирует настройки. Русские подписи и native tooltips описывают каждое поле. API keys, MODEL/CLIP/VAE сокеты и class IDs сохранены. Новые три поля добавлены в конец widget arrays. Отсутствующее поле означает `custom`: старые workflows не включают медленный RAM-профиль автоматически. Миграция смещённого `weight_placement` из присланной версии C сохранена.

Дополнительные примеры:

| Файл | Режим |
|---|---|
| `fl2va_ram_min_int8.ui.json` | INT8 H3 + INT8 Qwen, FSDP2, три GPU как пример |
| `fl2va_ram_min_all_sequence.ui.json` | INT8, все выбранные через `all`, FSDP2 + sequence |
| `ref2va_ram_min_fp16.ui.json` | Ref2VA, BF16 source → FP16 runtime, `all` |

Каждому UI соответствует `.api.json`. Во всех примерах сохранены audio decoding и исходные PNG после VAE. Для custom vLLM меняется только dropdown в соответствующем Config; ручной shim не нужен.

## Что именно остаётся в RAM и VRAM

`ram_min` — согласованная worker policy, входящая в session fingerprint. Изменение профиля требует соответствующей новой worker-сессии; clone исходного MODEL не меняется.

| Компонент | Поведение `ram_min` |
|---|---|
| Постоянные H3/Qwen параметры | CPU DTensor-шарды через штатный `CPUOffloadPolicy`; локальные строки читаются safetensors streaming |
| Активная группа параметров | H2D локального шарда и all-gather на GPU; `reshard_after_forward=True` у child, block и root |
| Гранулярность H3 | Attention и MLP — дочерние FSDP-группы. Нормы/modulation остаются в родительской группе |
| Гранулярность Qwen | Language attention и MLP отдельно, decoder parent отдельно. Vision остаётся поблочным |
| Explicit prefetch | Принудительно 0; служебные FSDP copy/all-gather buffers всё равно существуют |
| Prepared MLP weights | Отключены: нет полной подготовленной FP16-копии активного INT8 MLP для повторных chunks |
| INT8 | Хранение Linear — INT8 плюс metadata/scales; деквантизация активных строк. Не вся модель dense FP16 |
| Dense Linear | Конверсия/масштабирование весов по output rows; полный выход слоя всё равно выделяется |
| MLP | `auto` по согласованному бюджетy; override `manual/off` явно виден в отчёте |
| Worker conditioning-кэш | CPU LRU, `stage_cache_mib` на rank. Текущий RPC создаёт GPU inputs, удаляет их после результата |
| Spectrum | История принудительно CPU, если Spectrum включён. Его аппроксимация не меняется |
| Qwen embeddings-кэш | Отдельный существующий `cache_mib` в RAM основного процесса |
| SDPA | Q/head tiles, полный K/V. Нет обрезания attention или локального окна вместо полного attention |
| Flash/vLLM/Sage | Выбранный provider сохраняется. При семантическом fallback SDPA использует RAM tiling |
| VAE | Стандартные ноды и tiled decoding; перед VAE Release завершает H3 workers |

Данные активного слоя нельзя просто «выгрузить первыми»: GEMM/attention должны их читать. Inference без backward не хранит training-историю всех activations, поэтому обычный activation checkpointing здесь не решает пик. Экономия достигается уменьшением одновременно живущих тензоров и порций вычисления. Полные residual/output, глобальные K/V в token sequence, позиции/RoPE, некоторые native masks, root buffers и CUDA/NCCL остаются расходами VRAM.

`workspace_mib` — оценка временной работы, **не общий предел VRAM**. Даже минимальная строка SDPA содержит всю длину K. Кэш, NCCL, параметры активных вложенных групп, конверсия Q/K/V и workspace provider добавляются сверху. RAM-профиль не делает большие разрешения/длительности гарантированно помещающимися. CPU pinning тоже требует реальной RAM; при её дефиците выключите `pin_memory`, сравнив производительность.

Особенность сохранённого Qwen: INT8 embedding локально деквантизируется в floating CPU-шард при загрузке. INT8 Linear остаются INT8. Не заявляется, что каждый байт encoder остаётся квантованным; CPU footprint embedding больше размера его INT8-файла.

## Почему 90% могут оставаться и на трёх, и на пяти GPU

Без конкретных свежих rank reports нельзя назвать единственную причину. В коде есть следующие составляющие:

1. FSDP делит постоянные веса, но временно собирает активную группу. В `fsdp2` каждый rank также повторяет activations/compute одного примера.
2. В token sequence локальных Q/MLP меньше, но глобальные K/V остаются. Число GPU не делит все расходы на N.
3. `memory_reserved` — allocator cache, а `memory_allocated` — занятые tensors. `nvidia-smi` включает ещё CUDA/NCCL, драйвер и другие процессы. Нельзя складывать overlapping snapshots.
4. Длинные video/audio sequences, ref conditioning и Qwen vision увеличивают live activations. CPU offload весов сам по себе не перемещает эти тензоры.
5. Старый worker держал conditioning в GPU-кэше и создавал дополнительный clone после H2D. RAM-профиль убирает постоянный CUDA-кэш; лишний post-copy clone устранён для всех профилей.

Исправлена оценка тёплого allocator: native backend использует `driver_free + max(0, reserved - allocated - inactive_split)` как приблизительный бюджет. Это позволяет не падать искусственно до MLP-порций в один токен из-за кэша allocator. Для другого allocator неподтверждённые cache counters не считаются свободной памятью. Расчёт не является запретом запуска и не гарантирует отсутствие фрагментации/OOM.

Смотреть в rank JSON: `rpc_memory`, `stage_cache`, `memory_settings`, `memory_plan.allocator_budget`, `weight_memory`, `execution.units`, `cpu_memory`. `largest_unsharded_group_bytes` — размер одной группы, не весь одновременный пик. Вложенный путь удерживает root + block parent + активный child, а communication buffers учитываются отдельно в реальном CUDA peak.

## ATS: возможно, но это отдельный backend

ATS — аппаратная трансляция адресов. UVA означает общее адресное пространство; это не доказательство доступа GPU к произвольному CPU tensor. `cudaMallocManaged`, system-allocated unified memory и CPUOffloadPolicy — разные механизмы.

CUDA 12.4 описывает поддержку system-allocated unified memory на POWER9+Volta и требует проверять `pageableMemoryAccess` и связанные свойства. Обычная PyTorch CPU-аллокация от этого не становится CUDA tensor для произвольного matmul. Для модельного пути нужны управляемые allocations/совместимый allocator, корректные lifetimes/streams и проверка CUDA kernels/NCCL/FSDP. Глобальная замена allocator во всём ComfyUI здесь не сделана.

Практически возможен **отдельный opt-in managed-memory путь для выбранных activations или крупных embedding**, с CPU preferred-location, отдельным MemPool и проверкой коллективов. Это следующий эксперимент, не готовая функция этого выпуска. Сначала надо подтвердить на AC922 корректность, переходы страниц, pinned/pageable transfers и отсутствие патологического вытеснения. PyTorch предупреждает о цене page faults и двойных пересылок при переполнении GPU. ATS/UVM не исключают OOM в RAM, NCCL или неуправляемом CUDA workspace.

Старое значение `weight_placement=ats` сохранено для совместимости и честно подписано: **CPUOffloadPolicy + диагностика**, не ATS allocator. Нет ложного статуса «ATS inference PASS».

Добавлены `scripts/ats_memory_probe.cu` и `scripts/run_ats_probe.py`: в отдельных процессах по UUID проверяют kernel, читающий/изменяющий system `malloc` и `cudaMallocManaged`, целостность, cold/warm time и free VRAM. `malloc` вызывается GPU только при подтверждённом `pageableMemoryAccess`. Размер по умолчанию 64 MiB, намеренно не тест переполнения VRAM. Результат не доказывает модельный ATS или маршрут NVLink.

```bash
python scripts/run_ats_probe.py --gpus all --build --mib 64 --output reports/local-ats-memory.json
```

`--build` явно компилирует только этот небольшой probe во временную папку. Нужен ваш nvcc в PATH; torch/wheel не изменяются. Если compiler не понимает `-arch=native`, укажите поддерживаемый `--cuda-arch`, например `sm_70` на V100. Для готового binary используйте `--binary /ABS/ats_probe` вместо `--build`. В этом окружении: **NOT_RUN — нет CUDA GPU и nvcc**.

Источники, проверены 2026-09-29:

- [PyTorch 2.12 FSDP2 / CPUOffloadPolicy](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html)
- [CUDA 12.4 Unified Memory, раздел 19](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#unified-memory-programming)
- [PyTorch 2.12 CUDA MemPool и ограничения UVM](https://docs.pytorch.org/docs/2.12/notes/cuda.html#mixing-different-cuda-system-allocators-in-the-same-program)

## Ускорение и качество: что сравнивать

| Вариант | Память / скорость | Влияние на результат |
|---|---|---|
| `ram_min` | Ниже одновременные allocations; больше передач/малых GEMM/collectives | Полная математика сохранена, FP16 округления могут отличаться |
| `custom`, MLP 4096/8192/16384 | Более крупные GEMM и возможная подготовка весов, но выше пик | Не является приближением |
| `custom`, prefetch 1 затем 2 | Возможный overlap ценой VRAM; сравнивать warm sec/step | Не является приближением |
| Ваш FA/vLLM против SDPA | Меньше attention workspace возможно; нужен конкретный kernel benchmark | Проверить численные отклонения, masks/scale и PNG |
| FSDP2 + sequence | Меньше локальных token GEMM; дополнительные обмены | Тот же attention; выигрыша на короткой последовательности может не быть |
| GPU subset 3/5/all | Больше GPU не гарантирует ускорение, меняет топологию и collectives | При одинаковых inputs допустимы FP округления |
| Qwen `cpu_shards` + embedding cache | Повторы не требуют загрузки/кодирования; больше постоянной RAM | Ключ кэша учитывает модель, prompt/images и настройки |
| NUMA auto/bind, pinned on/off | Возможный выигрыш CPU transfers на AC922; только по измерениям | Не является приближением |
| Tiled video VAE | Уменьшает decode peak; слишком мелкие tiles медленнее | Проверять стыки/temporal overlap в PNG |
| INT8, SageAttention | Компактнее; не гарантированно быстрее на конкретном GPU | Отдельная погрешность квантования |
| Spectrum | Пропускает некоторые DiT проходы, когда хватает истории | Приближённый прогноз; риски деталей, движения, речи/audio |
| Меньше frames/разрешение/steps | Меньше работы, особенно tokens/attention | Меняется задача либо качество; это не бесплатная оптимизация |

Для качества сначала сравните одинаковые checkpoint/LoRA/seed/prompt/sampler/schedule/steps/resolution/frame count с Spectrum выключенным и неквантованным attention. Проверьте FP16 Safe, отсутствие NaN/Inf и корректность conditioning; не «лечите» их случайным CFG или clamp. BF16 source → FP16 runtime с FP32 islands может служить сравнением для INT8, но не считается BF16 compute на V100. Не обещается, что больше steps всегда лучше: используйте schedule конкретной модели и проверенного workflow. Новые Turbo/LoRA варианты требуют отдельной поддержки patch descriptors; этот выпуск их не добавляет.

Сохраняйте PNG до видеоencoder и audio отдельно: сжатие видео и ошибка attention — разные источники артефактов. Розовые кадры сами по себе не доказывают проблему shim. Оценка качества полной H3 в этом окружении **NOT_RUN**.

## Приёмка и сравнение на AC922

Все команды ниже запускаются Python действующего ComfyUI, из каталога плагина. Примеры GPU не валидация числа карт.

```bash
python scripts/diagnose.py --comfy /ABS/ComfyUI --output reports/local-environment.json
python scripts/probe_devices.py --gpus 0,1,2 --cpu-offload --memory-profile ram_min --attention-backend sdpa --reports reports/local-probe-ram
python scripts/benchmark_transfer.py --gpus all --numa-policy auto --mib 256 --iterations 20 --output reports/local-transfer.json
python scripts/accept_h3.py --comfy /ABS/ComfyUI --checkpoint /ABS/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --gpus 0,1,2 --precision int8_fp16 --memory-profile ram_min --workspace-mib 256 --attention-backend sdpa --output reports/local-h3-ram --debug-finite
python scripts/accept_qwen.py --comfy /ABS/ComfyUI --checkpoint /ABS/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors --gpus 0,1,2 --memory-profile ram_min --idle-policy cpu_shards --output reports/local-qwen-ram
```

`probe_devices` — CUDA/NCCL/tiny model проверка, не полный H3. `accept_h3` — pretrained forward с синтетическим conditioning, не качество готового видео. Окончательная проверка — обычный UI workflow с обоими native conditioning/VAE и SaveVideo/SaveImage.

Для сравнений используйте копию API workflow со своими checkpoint и параметрами. Генератор планов сам по себе задания не отправляет:

```bash
python scripts/benchmark_cases.py workflows/fl2va_ram_min_int8.api.json --axis memory --output-dir reports/local-memory-cases
```

Здесь `custom` сохраняет CPU offload исходного примера, `ram_min` добавляет меньшие группы/tiling/cache policy. Для сравнения GPU-resident FSDP и offload берите `custom` + `weight_placement=gpu`, затем `--axis offload`. Скрипт отклоняет бессмысленный offload/MLP benchmark внутри `ram_min`, где оба значения принудительно одинаковые. Явный запуск: добавьте `--submit --server http://127.0.0.1:8188`. ComfyUI cache-hit не является измерением новой генерации; скрипт отмечает его, cold/warm различайте по worker reports.

В паре сохраните: checkpoint revision/header hash, GPU UUID, requested/effective provider, seed, input, sampler/sigmas, steps, кадры/разрешение, offload и Spectrum. Сравните cold load, preprocess, warm sec/step, VAE, IPC, peak allocated/reserved каждого rank, суммарную CPU **PSS** (суммировать RSS mmap некорректно). Профилировщик `POWERSHARD_PROFILE=1` — отдельный измерительный запуск, не production fast path.

## Границы текущего результата

- Полный H3/Qwen32B, GPU FSDP2/CPU offload нового профиля, пользовательский wheel и ATS kernel на POWER9/V100 — **NOT_RUN_ON_AC922**.
- Возможны реальные OOM: активные embeddings/vision, большой full output/KV/native mask и kernel workspace не исчезают.
- Не весь encoder INT8: embedding-шарды преобразуются в floating, vision compute реплицирован.
- Нет нового allocator, автоматического повторного запуска после OOM с другим результатом, глобальной подмены ComfyUI или torch kernels.
- Результат numeric CPU tests не доказывает снижение VRAM, ускорение, качество видео или CUDA Tensor Core execution.
- Browser rendering в настоящем ComfyUI — NOT_RUN; JS-контракт панели и сериализации проверен в DOM harness.
