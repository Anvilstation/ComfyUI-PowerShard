# Дополнение 0.5.0rc1: RAM-профиль

Основа — существующий PowerShard и присланные изменения C, сохранённые отдельным commit `bb75376` после аудита. `DistributedConfig.memory_profile=ram_min` включает CPUOffloadPolicy, отключает explicit prefetch и prepared MLP storage. Meta → nested fully_shard → streaming local rows остаётся порядком загрузки. H3 attention/MLP и Qwen language self_attn/MLP — child FSDP units; norms/modulation принадлежат parent, root явно reshard=True. Это уменьшает группу весов, но parent/root и коммуникационные buffers остаются живыми при child forward.

Worker inputs отделены от bounded CPU conditioning cache; после H2D нет лишнего clone. Native allocator budget учитывает оценку свободных целых cache blocks; это не OOM gate. MLP и dense Linear ограничивают временную конверсию/вычисления. SDPA делится по Q/heads с исходными позициями masks и полным K/V; mathematically exact, кроме обычных FP/RNG различий. Custom providers не подменяются. Session fingerprint включает все новые поля; singleton/global monkey patch не добавлен.

UI schema — единый источник tooltips, readable labels и группированной панели; API keys/порядок прежних widgets сохранены. ATS direct-memory probe — отдельный subprocess, **не model allocator**. [Полное описание и ограничения](docs/RAM_ATS_UI_RU.md).

---

# Архитектура — 0.4.0

База: `f4a53c8a2e17c4f01741e65ef46deffc36531071`. Общий Session/subprocess/FileStore/NCCL runtime сохранён, добавлена роль qwen. Глобальный phase lock сериализует смену активной GPU-роли; idle CPU encoder можно сохранить без второго активного GPU generator. Worker policy входит в fingerprint; clone создаёт лёгкий host wrapper/configuration без копирования весов. ComfyUI classes/SDPA/sys.path/shim глобально не подменяются.

Новый `qwen_backend.py` переиспользует `load_local`/reshard/assert и plain INT8 representation. Native encoder остаётся владельцем tokenizer, vision/DeepStack, tags и выбора hidden state; FSDP root Entrypoint вызывает encode через hooks. Generated rotary buffer явно восстанавливается после meta materialization. Canonical offloaded tensors — локальные shards; N полных checkpoint CPU-копий не создаются по алгоритму загрузки.

MLP preparation живёт в context manager ровно один MLP; частично собранные scaled-half INT8 tiles очищаются и при исключении. Budget рассчитывается на границе RPC, согласуется между ranks в auto mode. FSDP wrapping остаётся блочным, chunks не создают дополнительные all-gather. Sequence communication K/V объединена, точность/порядок/padding сохранены.

Spectrum decision принимается **до root forward**: все ranks согласуют ACTUAL/FORECAST. Внешние gate modules установлены после загрузки FSDP children; на FORECAST их __call__/prefetch/all-gather не вызываются. Actual capture после последнего блока содержит только local target rows. Forecast восстанавливает текущий target один раз, native final heads выполняются всегда. Отдельный run/conditioning/timestep lane и bounded history предотвращают ghost state. Это приближённый opt-in с начальным Euler bridge; не имитация native forward кэшированным финальным видео.

Полная схема решений, формулы бюджета и ограничения: [PERFORMANCE_0_4.md](docs/PERFORMANCE_0_4.md). Все CUDA/FSDP/POWER9 результаты новых режимов — NOT_RUN; CPU suite — 195 PASS. Ниже сохранён исторический архитектурный аудит 0.3/0.2.

## Историческая база 0.3

Продолжена база 0.2.0 commit `9694964`; сохранены loader, representation, patcher и process runtime. Изменения: CUDA-visible inventory→UUID→N ranks, mesh `(N,)`, общий attention contract и внутренний adapter custom vLLM, предварительные изолированные probes и единая session policy. Число GPU/модель карты/CPU/SHA/Python build name/VRAM estimate не используются как admission gate. Отсутствие locality даёт предупреждение; фактические ошибки CUDA, формы, веса и API не игнорируются.

Новые детали: [MULTIGPU_ATTENTION.md](docs/MULTIGPU_ATTENTION.md). `attention_contract.py` задаёт математику, `attention_providers.py`/`vllm_adapter.py` — границы API, `attention_policy.py`/`attention_probe.py` — probes/fallback/fingerprint. `attention.py` остаётся worker-side H3 integration. Никакой глобальной подмены flash_attn/SDPA/классов.

CPU/native tests пройдены на ComfyUI `7a0b5eede3f9721c8faab290689893f36edc6d66` (2026-09-16); CUDA/provider/AC922 ещё NOT_RUN. Ниже сохранены обоснования предыдущего аудита и устройство backend; SHA 0.2 не являются ограничениями.

Продолжена существующая реализация на commit da461b8ab115e4e58a47074b55c70e357d3f9833. В 0.2 добавлены FP16 Safe, worker policy/clone isolation, CPUOffloadPolicy и NUMA; runtime/генератор не переписаны заново. Аудит: [FP16_FIX_AUDIT.md](docs/FP16_FIX_AUDIT.md).

Текущий ComfyUI проверен на 36da3ff763687eab86a35e1019995dd1fb369b0d (2026-09-15). SHA ниже — исторический аудит, не gate. source_guard проверяет фактические классы, методы и сигнатуры. Несовпадение версии не запрещает тест.

## Решение

Отдельный пакет PowerShard, родной `comfy.ldm.minimax.model.MiniMaxH3Model`, собственный локальный subprocess runtime. Raylight изучен по исходникам, но его код не копировался. Переиспользуются модель H3, packing, audio schedule, conditioning, ModelPatcher, sampler, CLIP и VAE ComfyUI. Существующие компоненты генератора не переписаны.

Причины отказаться от прямого форка Raylight:

1. Зафиксированный H3 adapter Raylight обращается к `PackedLayout(..., frame_count=...)`, которого нет в текущем конструкторе ComfyUI. Вызов `final_layer(h,t_emb,video_seg,audio_seg)` не передаёт новые `sigma`, `sample_sigmas`, `shifts`. Это конкретные API-разрывы, а не заключение по README.
2. В Raylight `usp_attn_forward` воспроизводит нормирование/RoPE, а `usp_dit_forward` воспроизводит весь старый `_forward`; при последующих изменениях audio scale, masks и layout такая копия расходится с upstream. PowerShard меняет только attention и границы блока; родной forward остаётся источником семантики.
3. Ray/xfuser/kernels увеличивают состав переносимых на ppc64le зависимостей. Доказательства, что Ray невозможно собрать на POWER9, не найдено; его сборка просто не нужна нашему управляющему слою. Используются стандартный subprocess и torch.distributed.
4. H3 имеет 56 heads; Ulysses degree=3 нельзя считать допустимым по умолчанию. Наш SP разделяет queries/tokens и допускает неравномерный последний rank.

## Проверенные исходники

| Компонент | SHA / версия | Что изучено |
|---|---|---|
| Raylight | `9a7c33d52b3d35e29f75ecff3c227de987f0d4cf`, 1.9.0 | `diffusion_models/minimax/xdit_context_parallel.py`, `comfy_dist/fsdp_utils.py`, `model_patcher.py`, `sd.py`, kitchen INT8 patch, Ray actors, requirements |
| ComfyUI | `683421b679f85e0f0fe00fc875640dfcd01103e9` | native H3, model detection/base, patcher/manager, sampler helpers, CLIP, audio/video nodes |
| MiniMax-AI/MiniMax-H3 | `d21241f0a4b3acbb34c97dae47fa417b7065e438` | configs, README, text/audio/video structure; не источник pruned single-file weights |
| Comfy Kitchen | `62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243`, 0.2.33 | ConvRot regular Hadamard, INT8 row scales, eager dequant, build flags |
| PyTorch | wheel `2.12.0+cpu`, source tag `v2.12.0` | FSDP2 param/group/collectives, CUDA/C++ configure gates |

Полный lock и hashes в `sources.lock.json`. Источники: [Raylight](https://github.com/komikndr/raylight), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3), [FSDP2](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html), [FSDP1](https://docs.pytorch.org/docs/2.12/fsdp.html).

## Аудит Raylight по требованиям

| Вопрос | Вывод по коду, не аппаратная сертификация |
|---|---|
| FSDP/FSDP2 | Используется `fully_shard`, bottom-up groups, meta placeholders, `reshard_after_forward=True`; root lazy-init специально обрабатывается |
| H3 FL2VA/Ref2VA | Есть adapter и регистрация; обнаружены указанные API-разрывы с audited ComfyUI |
| FSDP+SP | Общая схема есть; доказательства корректного H3 на degree=3/V100 в рассмотренном коде/тестах не получено |
| Quantization | INT8/FP8/NVFP4 kitchen patches, tensor subclass pre/post all-gather; это сложнее простой `.half()` |
| Loading | Meta/FSDP loading и state dict преобразования реализованы; пригодность расхода RAM для нашего сервера не измерена |
| Workers | Ray actors, отдельные worker environments; зависимости Ray и NCCL override требуют аудита установленного окружения |
| Release/restart | Есть явная очистка FSDP storage и обработка предыдущей ошибки инициализации; повторный H3 на нашем оборудовании не проверен |
| Volta | README содержит рекомендацию Yunchang, а не результат H3 3×V100. Это не принято за доказательство поддержки |
| License | Apache-2.0, Micko Lesmana и contributors; сохранена копия лицензии/NOTICE атрибуция |

## FSDP loader

`meta H3 → INT8 representation → instance FP16 patch → frozen parameters → block/root fully_shard → to_empty(local CPU/GPU shards) → safe_open.get_slice(local rows)`.

Все float/INT8 parameters — DTensor/Shard(0) между вызовами. Каждый main/refiner block и вспомогательные projections/final layer обёрнуты отдельно. Root содержит малые остаточные параметры. reshard_after_forward=True задан и root. В PyTorch 2.12 проверяются все _fsdp_param_groups, не только старый singular _fsdp_param_group. После RPC — reshard и assert SHARDED; probes проверяют также состояние непосредственно после hooks.

CPUOffloadPolicy(pin_memory=...) задаётся всем units. После meta wrapping to_empty materialize только CPU DTensor shards; get_slice читает локальные строки. Полных CPU checkpoint copies нет в production loader. Frozen CPU shards остаются каноническими: при inference не требуется копировать неизменённый полный блок D2H после каждого forward. Policy освобождает GPU materializations; H2D/prefetch происходят при следующем unshard. Реальный суммарный PSS/peak ещё не измерен.

Prefetch 0 (default) /1/2 задаётся через set_modules_to_forward_prefetch отдельным цепочкам refiner/main. Нет mesh 3×3. Дополнительные собранные blocks повышают peak; benefit overlap на AC922 NOT_RUN.

Patch и clone: [FP16_SAFE.md](docs/FP16_SAFE.md). Main хранит только proxy/config. Каждый worker применяет одну fingerprinted policy до fully_shard. Нет глобальных class patches или копирования весов при clone.

`Entrypoint.forward(command,args,kwargs)` оборачивает оба пути: обычный H3 `__call__` и `preprocess_text_embeds`. Поэтому предварительная обработка conditioning не обходит root FSDP hooks. Режим исполнения: `eval + inference_mode(False) + no_grad`, без master weights, optimizer и backward. Устойчивость этого пути на CUDA ещё должна пройти probe.

Loader проверяет ключи, формы, finite ranges, формат quant metadata. Tied parameters сейчас отклоняются, поскольку в подтверждённых H3 headers их mapping отсутствует. Никакого молчаливого раздваивания shared weights.

## INT8 ConvRot

В исходном checkpoint: `weight` I8, `weight_scale` F32 `[out,1]`, `comfy_quant` U8 JSON. Подтверждён формат `int8_tensorwise`, `convrot=true`, group=256. Runtime читает metadata **каждого** integer Linear, а не угадывает формат по имени.

PyTorch 2.12 `_fsdp_param.py` не применяет mixed precision к non-floating parameters; `_fsdp_param_group.py` проверяет однородность dtype только trainable parameters; `_fsdp_collectives.py` поддерживает смешанные dtype через byte communication. Поэтому реализованы обычные замороженные `nn.Parameter(int8)` и `nn.Parameter(scale)`, без kitchen tensor subclass. CUDA/NCCL совместимость этой комбинации остаётся NOT_RUN и проверяется обязательным tiny INT8 FSDP probe.

После all-gather блока dequant работает по 256 выходных строк. Используется регулярная симметричная ортогональная Hadamard ConvRot, **не** стандартная Sylvester H2. Сначала I8×row_scale в FP32, затем обратная rotation, FP16 GEMM. Временный вес не кэшируется. Хранение INT8 сохраняется, активации не квантуются. Совпадение dequant с eager Comfy Kitchen проверено на CPU.

## Sequence parallelism

Один mesh `(N,)` совместно используется FSDP и sequence collectives. Перед block0 полный packed stream делится на contiguous rows по `torch.chunk`-совместимым границам. Для каждого rank срезаются RoPE и modality/timestep segments, включая tensor-valued mask rows. Text, condition, reference, audio, video остаются частью родного PackedLayout. При пустом token rank — согласованный FSDP-only fallback на всех выбранных GPU.

Attention: локальные Q/K/V, all-gather глобальных K и V, полный точный online-softmax attention для локальных Q. Padding существует только в communication buffers и удаляется до softmax. Фиктивные tokens не влияют на результат. Heads не делятся, 56%3 не имеет значения. После последнего блока результаты собираются в исходном порядке; final layer и output packing родные.

Основные attention/MLP блока распределены по tokens; embedding, text refiner, adaLN и final projection частично/полностью повторяются. K/V реплицируются временно. Это не Ring Attention и не Ulysses; отсутствие head-divisibility ограничения оплачивается дополнительной K/V памятью/communication. Аппаратное ускорение не доказано.

## ComfyUI bridge

`MODEL = PowerShardPatcher(RemoteH3(DiffusionProxy))`. Это наследники реальных ComfyUI ModelPatcher/BaseModel, с model_sampling, audio scaling, latent packing, extra_conds, clone и options. Proxy не содержит parameters генератора. Memory manager видит оценку локальной доли, а переносит только малые служебные tensors. Dynamic VRAM генератора отключён.

Внутренний host PREPARE_SAMPLING wrapper проверяет все ветки conditioning до RPC. Он выполняется в host и удаляется из сериализуемых options по идентичности функции. Сторонние wrappers/callbacks, ControlNet и LoRA отклоняются. ModelSampling sigma shift сохраняется как supported object patch. RPC передаёт только структуры JSON и safetensors, без pickle функций. PackedLayout восстанавливается из всех native payload fields в worker.

## Runtime и жизненный цикл

N `subprocess.Popen`, не fork CUDA. Private temp directory, pipes управления, FileStore rendezvous, UUID mapping. NCCL сам открывает необходимые transport endpoints; дополнительного внешнего API управления нет. Последовательные команды, per-rank mailboxes, sequence IDs и общий deadline. Noise/sigma/conditioning вычисляет host, все ranks получают один snapshot.

Ошибка/отмена завершает всю свою сессию: graceful shutdown, затем terminate/kill только записанных PIDs. Новый RPC создаёт чистую группу. Workers не пересоздаются между diffusion steps. По умолчанию sampling cleanup завершает сессию перед VAE. Отдельная Release node делает порядок фазы явным. Освобождение кэшированного loader не уничтожает его конфигурацию, следующая генерация может создать workers снова.
