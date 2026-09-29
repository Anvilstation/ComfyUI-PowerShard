# ComfyUI-PowerShard

**Распределённый MiniMax H3 в обычном графе ComfyUI.** PowerShard запускает одну генерацию на выбранных CUDA GPU. FSDP2 шардирует веса генератора, а родные H3 conditioning, sampler и video/audio VAE остаются частью workflow. Для V100 предусмотрен FP16-safe путь; INT8 ConvRot не разворачивается заранее в постоянную FP16-копию.

[English](README.md) · [Совместимость](COMPATIBILITY.md) · [Результаты проверок](BENCHMARKS.md)

## Возможности

| Область | Реализация |
|---|---|
| Выбор GPU | Любое непустое подмножество видимых CUDA устройств: `0`, `1,3,5`, `5,2,0` или `all`. Порядок сохраняется. |
| Шардирование H3 | По одному worker на GPU; FSDP2 FULL_SHARD собирает параметры активных блоков и снова шардирует их после forward. На одной GPU меж-GPU шардирования нет. |
| MiniMax H3 | FL2VA и Ref2VA Pruned, BF16 checkpoint с FP16 runtime path и INT8 ConvRot. Worker-side FP16 Safe оставляет опасные участки в FP32. |
| Attention | `auto`, PyTorch SDPA, FlashAttention, пользовательский `vllm_flash_attn`, SageAttention и reference/math. Недоступный provider даёт объяснимый fallback при `allow_fallback=true`. |
| Распределение вычислений | Отдельный `fsdp2_sequence` делит token/query работу; обычный `fsdp2` прежде всего экономит память весов. |
| Qwen3-VL-32B CLIP | Нода для H3 text/vision conditioning с FP16/INT8, распределённым размещением, CPU offload и ограниченным RAM-кэшем. |
| Память | CPU offload локальных FSDP-шардов, управляемый prefetch и профиль `ram_min` для агрессивной выгрузки весов в RAM. Он может быть медленнее. |

Пакет **не заменяет** ComfyUI, CUDA, PyTorch, NCCL или установленный пользовательский wheel. Опциональные attention-библиотеки загружаются по необходимости. Веса моделей не входят в репозиторий.

## Быстрый старт

Используйте Python установленной ComfyUI и сохраните копию существующей ноды перед обновлением. Поместите проект в `ComfyUI/custom_nodes/ComfyUI-PowerShard`. Проверьте окружение и план зависимостей до установки:

```bash
python scripts/diagnose.py --comfy /ABS/ComfyUI --output reports/local-environment.json
python scripts/check_source_requirements.py
python scripts/dependency_plan.py --output-dir reports/local-dependency-plan
```

При необходимости установите только недостающие runtime-зависимости через существующий Python. Не обновляйте torch или пользовательский CUDA wheel автоматически. Инструкции для [ppc64le](docs/INSTALL_PPC64LE.md) и [x86_64](docs/INSTALL_X86_64.md).

Запуск из каталога PowerShard:

```bash
bash scripts/launch_comfy.sh /ABS/venv/bin/python /ABS/ComfyUI
```

Откройте **`workflows/fl2va_ram_min_int8.ui.json`** для профиля «Минимум VRAM» либо **`workflows/fl2va_fp16.ui.json`** для базового FP16. Выберите локальные H3/Qwen/VAE checkpoints и свои GPU в Distributed Config. `0,1,2` в примерах — обычный список, не ограничение проекта. MODEL проходит через **PowerShard MiniMax H3 FP16 Patcher** к родному guider/sampler.

До загрузки больших весов проверьте CUDA/NCCL на выбранных картах:

```bash
python scripts/probe_devices.py --gpus 0,1,2 --attention-backend sdpa --reports reports/local-probe
```

[Checkpoints](models/CHECKPOINTS_RU.md) · [Другие workflows](workflows/) · [RAM/ATS/UI: команды и настройки](docs/RAM_ATS_UI_RU.md)

## Статус проверки

Версия **0.5.0rc1**: локально на Linux x86_64 / Python 3.11.16 / torch 2.12.0+cpu прошли 241 основной тест, 58 регрессий аудита и отдельный benchmark CLI тест. Это проверки CPU-контрактов, native H3/Qwen малой размерности, интерфейса и математики; они **не доказывают** работу CUDA/FSDP на V100.

Пользователь сообщил об успешных запусках прежней серверной версии на AC922 с тремя V100: `fsdp2` и `fsdp2_sequence` с vllm-FA, FA и SDPA. Новый профиль `ram_min`, Qwen32B, Spectrum, пользовательский wheel и реальные VRAM/скорость **не проверены нами на AC922**. Полная pretrained H3 генерация этим выпуском здесь **NOT_RUN**. Измеренного ускорения или гарантии отсутствия OOM нет.

Точные статусы: [отчёт 0.5.0rc1](reports/ram-min-2026-09-29/REPORT_RU.md), [benchmark](BENCHMARKS.md), [ограничения](KNOWN_LIMITATIONS.md). Spectrum — отдельный экспериментальный режим с приближением; ATS allocator для модели пока не реализован.

## Документация

- [Устройство backend и FSDP](ARCHITECTURE.md)
- [Совместимость ComfyUI и режимов](COMPATIBILITY.md)
- [GPU, attention и пользовательский vllm wheel](docs/MULTIGPU_ATTENTION.md)
- [Память, ATS, интерфейс, скорость и качество](docs/RAM_ATS_UI_RU.md)
- [FP16 Safe](docs/FP16_SAFE.md)
- [История прежних выпусков и подробные команды](README_HISTORY_RU.md)

Лицензия проекта: [GPL-3.0](LICENSE). Лицензии сторонних компонентов сохранены в [licenses](licenses/).
