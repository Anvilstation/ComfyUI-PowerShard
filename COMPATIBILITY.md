# Матрица дополнений 0.5.0rc1

| Комбинация | Реализация | Фактическая проверка |
|---|---|---|
| H3 FP16 Safe + RAM tiling + SDPA | Да, DiT/refiner/condition/video/audio | PASS, native уменьшенная модель CPU |
| Qwen CLIP + RAM tiling, dense/INT8, text/image/video | Да | PASS, native уменьшенная модель CPU |
| H3/Qwen RAM nested FSDP + CPU offload | Код и CLI есть | NOT_RUN: нет CUDA |
| 1..N выбранных GPU | Сохранено | Python device/runtime contract PASS; hardware NOT_RUN |
| Flash/vLLM/Sage + RAM profile | Выбор provider сохранён, semantic fallback использует tiling | CUDA NOT_RUN; пользовательский POWER9 wheel NOT_RUN |
| MODEL clone, sampler, extra_conds | Сохранено | CPU native tests PASS |
| Старые widget arrays + панель настроек | Совместимость и C migration | JS DOM contract PASS; настоящий browser NOT_RUN |
| ATS/UVM H3 allocator | Не реализован | НЕ называется offload/diagnostic PASS |

Неизвестная версия/архитектура сама по себе не запрещает выполнение. Обнаруженные отсутствующие API, некорректные tensors и реальные ошибки kernels по-прежнему ошибки. Подробности: [RAM/ATS/UI](docs/RAM_ATS_UI_RU.md).

---

# Совместимость 0.4.0

ComfyUI `7a0b5eede3f9721c8faab290689893f36edc6d66`, Python3.11.16/torch2.12.0+cpu: **195 PASS**. SHA и версии только диагностические, сохранён capability detection. CUDA/POWER9 для новых функций не доступны. Сообщённые пользователем серверные успехи — USER_REPORTED.

| Новая комбинация | Реализованный путь | Проверка здесь |
|---|---|---|
| Native H3 → Safe → BasicGuider/SamplerCustomAdvanced | Старый adapter/runtime сохранён, run JSON на границе sampler | CPU native/subprocess PASS |
| MLP manual/auto/off + INT8/F16 | Тот же MLP, bounded ephemeral weight preparation | CPU numerical PASS; V100 speed NOT_RUN |
| Qwen Loader → CLIP → native scheduled conditioning | Наследник CLIP, настоящий tokenizer/tags/hidden layer; lightweight clone/cache | CPU tiny native PASS |
| Qwen text/image/reference-video | Native processor/vision/DeepStack | CPU tiny BF16-derived FP16 + INT8 PASS; pretrained NOT_RUN |
| Qwen reference audio | Native текстовые labels; audio waveform идёт по H3 audio conditioning, не Qwen | Native код сохранён; полный pipeline NOT_RUN |
| Qwen FSDP2 / CPUOffloadPolicy / idle CPU shards | Блочные groups и streaming local slices в общей worker role | CUDA NOT_RUN |
| Qwen fsdp2_sequence | Local query/MLP, global KV+hidden each layer; vision compute replicated | CPU math/slices PASS; NCCL/скорость NOT_RUN |
| Qwen custom attention | Общий selector, GQA/vision semantic fallback с честными counters | CPU contracts PASS; kernel NOT_RUN |
| Spectrum + deterministic native Euler | Worker external FSDP gates, shared decision, native final AV heads | CPU native/subprocess PASS; FSDP CUDA NOT_RUN |
| Spectrum + другие samplers / s_churn | Warning, ACTUAL обычного backend, без скрытой подмены solver | Capability fallback CPU PASS; прочие sampler executions NOT_RUN |
| Spectrum + sequence | Gates вокруг исходных sequence FSDP blocks, local target history | Local partition CPU PASS; combined CUDA NOT_RUN |
| Spectrum Turbo/few-step | Euler4 даёт 0 forecasts при default warmup/tail; Turbo-specific solver bridge отсутствует | CPU policy PASS; trained Turbo weights/качество NOT_RUN |
| LoRA/ControlNet/custom model wrappers | Отсутствовали в полученной базе 0.3; новая поддержка не объявляется | UNSUPPORTED, серверный diff не предоставлен |
| Release перед native VAE | Закрывает активную DiT, optional сохранение idle CPU Qwen/cache | Python/native lifecycle PASS; real VAE/GPU NOT_RUN |

Старые class IDs, JSON и порядок прежних widgets сохранены. Новая Qwen-нода дополняет прежний native text loader. Нет искусственного ограничения 3/6 GPU, V100/AC922, Python build name или estimates VRAM. Пустой/несуществующий GPU, повреждённый checkpoint, несовместимый математический контракт, actual CUDA/NCCL ошибки по-прежнему являются ошибками.

Подробные ограничения, точные dtypes и команды: [PERFORMANCE_0_4.md](docs/PERFORMANCE_0_4.md).

## Историческая матрица 0.3

Ниже состояние прежней версии; строка старого CLIP loader не описывает новый distributed Qwen 0.4.

Новая база проверена на ComfyUI `7a0b5eede3f9721c8faab290689893f36edc6d66`, Python3.11/torch2.12 CPU: 147 PASS. GPU и custom wheel не доступны. Выбор N=1..число видимых CUDA GPU, без hardware/version gates; ppc64le и x86_64 используют одни файлы. Подробная attention matrix/ограничения: [MULTIGPU_ATTENTION.md](docs/MULTIGPU_ATTENTION.md).

Все шесть вариантов attention имеют реальную worker integration или явный fallback. Custom CUDA policy проверяет один FP16/head_dim профиль; сложные masks/GQA/window/ALiBi/dropout/rectangular causal идут через семантически точный SDPA/math, пока не сертифицированы на данном kernel. Sage включается только явным выбором. `allow_fallback=false` делает недоступный явно запрошенный путь ошибкой. Ни один неизвестный GPU не исключается по имени.

Версия/SHA ComfyUI — только запись диагностики. Допуск определяется capabilities ModelPatcher, native MiniMaxH3Model/DiTBlock/MLP, model_base, sampler и patcher_extension. Проверено на актуальном доступном checkout 36da3ff763687eab86a35e1019995dd1fb369b0d. Номер версии не служит причиной пропуска теста.

«Реализовано» не означает аппаратный PASS. Реальные checkpoints/CUDA/POWER9 ниже NOT_RUN.

| Комбинация | Execution path | Фактическая проверка |
|---|---|---|
| FL2VA/Ref2VA Pruned BF16 → FP16 Safe | Native H3, FP32 condition/residual, scaled half GEMMs | Tiny native CPU PASS; полный meta по header PASS |
| INT8 ConvRot + Safe | I8 Parameters/scales, temporary row dequant; group 256 | Dequant vs Kitchen и tiny H3 AV forward CPU PASS |
| MODEL → FP16 Patcher → extra_conds | Host structural clone + worker policy + FP32 preprocess | Реальный CPU subprocess PASS, input 100000 |
| MODEL → BasicGuider/BasicScheduler → SamplerCustomAdvanced | Native CFGGuider и patcher_extension, host callback/x0 | CPU subprocess tiny H3 PASS, оба AV outputs |
| Обычный KSampler Euler/simple CFG=1 | Native sampler + proxy | CPU subprocess PASS; seed 5/6/5, изменённые conditioning |
| Остальные samplers | Сохранён native interface; патчи sampler проходят whitelist | NOT_RUN; несовместимые patches отклоняются |
| N выбранных GPU FSDP2 FULL SHARD | Один mesh (N,), block/root reshard; N=1 без inter-GPU sharding | NOT_RUN CUDA; Python partition/runtime tests PASS |
| FSDP2 + CPUOffloadPolicy | CPU local shards + pinned/pageable option | NOT_RUN_ON_AC922 и NOT_RUN CUDA x86 |
| FSDP2 prefetch 0/1/2 | Отдельные main/refiner chains | API реализован, GPU memory/overlap NOT_RUN |
| FSDP2 + sequence | Query/token split + global KV, uneven rows | Attention math CPU PASS; три-rank H3/SP NOT_RUN |
| Text/first-last frame/Ref2VA image/audio conditioning | Native payload/packing; FP32 stream | Tiny text/masks/keyframes/image/audio CPU PASS |
| Reference audio/video полный pipeline | Native payload сохраняется | Real encoders/VAE/conditioning NOT_RUN |
| CLIP | Родной H3 Qwen3-VL, cpu_fp32 / native_offload | Loader audited; real encoder NOT_RUN |
| Video/audio VAE, tiled decode, SaveVideo | Родные ноды; workers release перед decode | Реальные веса/output files NOT_RUN |
| Clone/cache, отключение patch | Новый policy/session, изолированный proxy tree | CPU subprocess PASS; исходный MODEL не изменяется |
| LoRA/ControlNet/произвольные wrappers | Не реализовано | Явная ошибка до RPC/collectives, не silent ignore |
| POWER9 Python3.11 / CUDA12.4 / torch2.12 | Общая кодовая база, optional affinity | NOT_RUN_ON_AC922 |
| x86_64 Python3.11 / V100 sm_70 | Общая кодовая база | CPU only PASS; CUDA NOT_RUN |
| Более новые GPU | Тот же переносимый FP16 path | NOT_RUN; BF16/FP8/fused kernels не включаются автоматически |

Patcher принимает MODEL именно из PowerShard H3 Loader. Он не пытается патчить произвольный native/чужой MODEL. Это явное ограничение, а не фиктивный socket.

tests/cpu_contract_worker.py — только тестовая CPU точка входа, один subprocess без NCCL/FSDP. Production powershard.worker получает N из выбранных UUID. Нельзя переносить PASS тестового transport на аппаратную приёмку. Таблица native pipeline содержит сохранённые результаты 0.2, повторно проверенные в CPU suite 0.3.
