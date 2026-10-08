> Архив первоначального пакета 0.4.0; не является текущей GPU-приёмкой.

# ComfyUI-PowerShard 0.4.0 — Qwen CLIP, память и Spectrum

Продолжена существующая версия 0.3.0 (`f4a53c8`). Сохранены произвольный выбор GPU, attention adapters, FSDP2/sequence, FP16 Safe и INT8 ConvRot. Установка не заменяет PyTorch, CUDA/NCCL, ComfyUI или пользовательский vllm_flash_attn wheel.

Добавлены:

- `mlp_chunk_mode=manual/auto/off`, подготовка активного MLP один раз на chunks при достаточном бюджете; FP16 Safe не отключается вместе с chunking.
- `memory_policy=auto`: активации/коммуникации/reserve учитываются до prefetch и MLP workspace. Реальный OOM не маскируется прогнозом.
- **PowerShard H3 Qwen Loader → CLIP**: native truncated Qwen3-VL-32B, FP16/INT8, FSDP, отдельный sequence-путь, CPUOffloadPolicy, idle CPU shards и bounded conditioning cache.
- **PowerShard Spectrum → MODEL**: выключен по умолчанию, прогноз в workers, согласованное решение ranks до FSDP, local target history, native audio/video heads. Начальный bridge — deterministic Euler; другие samplers выполняют обычный FSDP с предупреждением.
- Один K/V collective вместо двух в H3 sequence, устранён повторный обход metadata пакетов на каждом RPC; отчёты и парные benchmark cases.

**Проверено локально: 195 PASS, 0 FAIL/skip**, Python 3.11.16, torch 2.12.0+cpu, x86_64. Настоящие native ComfyUI H3/Qwen классы, sampler/CLIP/clone/cache, CPU subprocess, INT8 math и sequence-разрезы. **Полная H3 generation, новые CUDA/FSDP/Qwen/Spectrum режимы, реальные VRAM/PSS и POWER9/V100 custom wheel — NOT_RUN / NOT_RUN_ON_AC922.** Ваши успешные AC922 FSDP/sequence + FA/vllm-fa/SDPA записаны как USER_REPORTED, а не наши замеры.

Первый новый граф: `workflows/fl2va_qwen_int8_offload.ui.json` (25 кадров, Spectrum отсутствует). Для сохранённого прежнего encoder остаются все старые workflows. Выберите свои GPU и локальные checkpoints. Из каталога проекта:

```bash
bash scripts/launch_comfy.sh /ABS/venv/bin/python /ABS/ComfyUI
```

Подробные ответы про скорость, MLP, 48 ГБ, Qwen, Spectrum и команды: **[PERFORMANCE_0_4.md](docs/PERFORMANCE_0_4.md)**. Фактическая приёмка и изменённые файлы: [IMPLEMENTATION_REPORT_RU.md](IMPLEMENTATION_REPORT_RU.md), [BENCHMARKS.md](BENCHMARKS.md).

Обновление накладывайте на сохранённую копию своих изменений: архив содержит исходники, Git bundle и delta относительно `f4a53c8`. Если серверная версия отличается, сначала `git apply --check` для delta. В доступной базе нет LoRA descriptors; серверные доработки LoRA не предоставлены, их нельзя подтвердить или перезаписать вслепую.

## Справка о сохранённой версии 0.3

Ниже историческая инструкция 0.3. Новые возможности/статусы выше и в PERFORMANCE_0_4 дополняют её; старый encoder по-прежнему доступен отдельно.

GPU: любое непустое подмножество видимых CUDA устройств, `0`, `1,3,5`, `5,2,0`, `all`; без максимума 3 или 6. Config node сохраняет старые поля и добавляет кнопку выбора карт, `attention_backend` и `allow_fallback=true`.

Attention: **auto / sdpa / flash_attn / vllm_flash_attn / sageattention / math**. Внутренний adapter использует пользовательский kernels-пакет без глобального shim и без установки всего vLLM. Выбор применяется к настоящим worker-side H3 DiT и token refiner. FSDP, FP16 patch, INT8 representation, CPU offload и prefetch сохранены.

Начните с [инструкции 0.3.0](docs/MULTIGPU_ATTENTION.md): семантика GPU IDs, миграция shim, fallback, установка локального wheel и точные команды. Первый граф — `workflows/fl2va_all_sdpa.ui.json`; custom vLLM + offload — `workflows/fl2va_all_vllm_offload.ui.json`. Оба сохраняют PNG до видеокодирования. Старые графы остаются в каталоге.

**Фактически: 147 CPU PASS**, включая native H3 и ComfyUI SamplerCustomAdvanced на commit `7a0b5eede3f9721c8faab290689893f36edc6d66`; Python 3.11.16 / torch 2.12.0+cpu. Реальные CUDA/NCCL/FSDP, POWER9/V100 custom wheel, pretrained H3 generation, rank VRAM и speedup — **NOT_RUN**. CPU/Gloo transport — BLOCKED (Operation not permitted), не PASS.

Исходная база — сохранённый 0.2.0 commit `9694964`. Более поздний серверный код не предоставлен; в этой базе **нет LoRA descriptors**. Они не удалялись, но их совместимость нельзя подтвердить без серверного diff. Перед обновлением сохраните свои изменения. Подробнее: [отчёт](IMPLEMENTATION_REPORT_RU.md), [ограничения](KNOWN_LIMITATIONS.md).

## Сохранённые возможности базового pipeline

Доработка существующего PowerShard. Сохранены subprocess runtime, native ComfyUI H3 adapter, FSDP2 и INT8 ConvRot loader; core ComfyUI не изменяется.

Главная аппаратная цель ещё не принята: нет NVIDIA GPU, AC922 и локальных реальных checkpoints. Исторические результаты 0.2.0 отделены от новых в [BENCHMARKS.md](BENCHMARKS.md).

## Изменения

- **PowerShard MiniMax H3 FP16 Patcher**, MODEL → MODEL: клонирует wrapper/configuration, не веса. Patch применяется к H3 внутри каждого worker **до FSDP**.
- condition_proj, residual, нормализация, SiLU/gating и native safety islands работают FP32. Тяжёлые attention/MLP GEMM используют математически масштабированные FP16 operands и FP32 восстановление результата. Нет clamp/nan_to_num и полной FP32 модели.
- INT8 weights/scales остаются шардированными; деквантуется только порция строк активного Linear.
- cpu_offload=false по умолчанию. При включении используется CPUOffloadPolicy для **локальных CPU шардов**, не трёх CPU копий.
- prefetch_blocks=0/1/2, pin_memory=true/false, numa_policy=none/auto/bind.
- SHA/version gating ComfyUI удалён. Проверяются методы/сигнатуры; несовпадение версии не запрещает испытания. Старое поле allow_unverified сохранено для графов и больше не блокирует запуск.
- Нет GPU→CPU чтения finite/max в каждом блоке. Один deferred finite check на границе RPC; debug_finite добавляет GPU flags по модулям.

Подробности: [FP16_SAFE.md](docs/FP16_SAFE.md), [аудит fixes](docs/FP16_FIX_AUDIT.md).

## Безопасная установка

Используйте **существующий Python 3.11 с вашим CUDA PyTorch**. CPU wheel из локального отчёта не предназначен для V100! Поместите проект в ComfyUI/custom_nodes/ComfyUI-PowerShard, сохранив пользовательские изменения. Из этой папки:

```bash
python scripts/diagnose.py --comfy ../.. --output reports/local-environment.json
python scripts/check_source_requirements.py
python scripts/dependency_plan.py --output-dir reports/local-dependency-plan
```

Просмотрите план. Только если он устраивает:

```bash
python -m pip install -c reports/local-dependency-plan/protected-constraints.txt -r requirements-runtime.txt
```

PowerShard добавляет safetensors==0.6.2, остальные зависимости берёт из существующей ComfyUI. Не запускайте pip install -U или весь upstream requirements поверх кастомного стека. Инструкции: [ppc64le](docs/INSTALL_PPC64LE.md), [x86_64](docs/INSTALL_X86_64.md).

## До реальных весов

```bash
python scripts/probe_three.py --gpus 0,1,2 --timeout 120
python scripts/probe_h3_cuda.py --comfy ../.. --gpus 0,1,2
python scripts/probe_h3_cuda.py --comfy ../.. --gpus 0,1,2 --cpu-offload
python scripts/benchmark_transfer.py --gpus 0,1,2 --numa-policy none
```

Первый probe: CUDA compute, NCCL broadcast/all-gather/all-reduce, frozen FP16/INT8 FSDP2, реконструкция tiny weights и пять forward без backward. Он автоматически выполняется при старте H3 workers до больших весов. Второй/третий проверяют tiny native H3 + FP16 Safe, оба storage formats, preprocess и AV forward. Это не реальные pretrained weights. UUID из диагностики предпочтительнее индексов; порядок один для всех ranks.

## Первый workflow

| Каталог ComfyUI | Локальный файл |
|---|---|
| models/diffusion_models | minimax_h3_fl2va_pruned_bf16.safetensors |
| models/text_encoders | qwen3vl_32b_minimax_h3_bf16.safetensors |
| models/vae | minimax_h3_video_vae_fp16.safetensors |
| models/vae | minimax_h3_audio_vae_fp32.safetensors |

INT8 generator: minimax_h3_fl2va_pruned_int8_convrot.safetensors. Ref2VA, revisions и hashes: [CHECKPOINTS_RU.md](models/CHECKPOINTS_RU.md). extra_model_paths.yaml поддерживается через native folder_paths. Скачивание — отдельная явная операция scripts/download_checkpoint.py с --download, только одного файла; оригиналы не меняются.

Точная команда из каталога PowerShard, замените два пути:

```bash
bash scripts/launch_comfy.sh /ABS/venv/bin/python /ABS/ComfyUI
```

Эквивалент из каталога ComfyUI:

```bash
python main.py --listen 127.0.0.1 --disable-dynamic-vram --use-pytorch-cross-attention
```

Откройте workflows/fl2va_fp16.ui.json. Уже подключены Loader → FP16 Patcher → BasicGuider/BasicScheduler → SamplerCustomAdvanced → Release → оба VAE → SaveVideo. Выберите файлы и желаемые GPU; 0,1,2 — только прежний default. Оставьте patch enabled и fp16_safe.

INT8: fl2va_int8.ui.json. Offload: fl2va_fp16_offload.ui.json или fl2va_int8_offload.ui.json. Все графы используют одну кодовую базу, API-версии лежат рядом.

Первый граф: 768×768, 5 кадров, 20 steps, Euler/simple, seed 44, audio включено. Это короткий shape smoke, не проверка обученного качества длительного ролика. Реальный peak неизвестен. Encoder cpu_fp32 требует примерно 96 GiB только весов плюс рабочую RAM; отдельно есть native encoder offload, аппаратно NOT_RUN. Offload генератора не означает offload энкодера.

Прямой тест реального генератора после probes:

```bash
python scripts/accept_h3.py --comfy ../.. --checkpoint ../../models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --precision int8_fp16 --gpus 0,1,2 --debug-finite --lifecycle
```

FP16 Safe включён в CLI по умолчанию. Offload: --cpu-offload --prefetch-blocks 1. Большой condition stress: --text-magnitude 100000; это синтетический input, не естественный prompt.

Запуск API графа против уже работающей локальной ComfyUI:

```bash
python scripts/run_api_workflow.py workflows/fl2va_int8.api.json --submit --output reports/local-e2e.json
```

Без --submit — только план. Логи: ComfyUI/output/powershard. Для всех ranks должны совпадать patch fingerprints и присутствовать shard rows/device, sharded-after-forward, memory counters.

## Измерения и ограничения

fsdp2 при N>1 экономит память весов, но повторяет вычисление одного примера на ranks. fsdp2_sequence — отдельный экспериментальный query/token-sharding backend на том же выбранном наборе N GPU, без ограничения heads % N. При N=1 меж-GPU распределения нет. Ускорение **не измерено**.

[Совместимость](COMPATIBILITY.md), [приёмка и NUMA/benchmark](docs/ACCEPTANCE.md), [ограничения](KNOWN_LIMITATIONS.md). Ни один CPU PASS не является доказательством трёх GPU FSDP.
