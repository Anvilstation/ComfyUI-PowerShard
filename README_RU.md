# ComfyUI-PowerShard

**Распределённая генерация MiniMax H3 и Wan в ComfyUI, с поддержкой LTX.** Одна генерация выполняется на выбранном наборе CUDA GPU: FSDP2 шардирует веса, а token / Ulysses распределяет вычисления по токенам. В графе остаются родные conditioning, sampler и VAE-ноды ComfyUI.

[English](README.md) · [Последний релиз](https://github.com/Anvilstation/ComfyUI-PowerShard/releases/latest) · [Совместимость](COMPATIBILITY.md) · [Лицензия](LICENSE)

## Возможности

| Область | Что входит в релиз |
|---|---|
| MiniMax H3 | FL2VA / Ref2VA, FP16 Safe, INT8 ConvRot, распределённый Qwen3-VL для conditioning. |
| Wan 2.1 / 2.2 | T2V, I2V, A14B high/low MoE, TI2V-5B, LoRA и специализированные варианты видео и управления. |
| LTX | Загрузчики LTX-Video / LTX-AV, аудио/видео workflows, LoRA / IC-LoRA, распределённый Gemma. |
| Выбор GPU | Любой непустой набор устройств, видимых через CUDA: `0`, `0,1,2`, `5,2,0` или `all`. |
| Распределение | FSDP2 делит веса; token / Ulysses распределяет работу с последовательностями. |
| Память | Шарды в VRAM, CPU offload, сохранение весов в RAM между задачами и экспериментальный ATS. |
| Attention и точность | Выбор attention provider, FP16 Safe для V100, настройка квантованного хранения с учётом модели и установленной ComfyUI. |
| Готовые графы | MiniMax в [workflows](workflows/), Wan в [workflows_wan](workflows_wan/), LTX в [workflows_ltx](workflows_ltx/). |

## Установка

Клонируйте репозиторий в установленную ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Anvilstation/ComfyUI-PowerShard.git
```

Либо скачайте ZIP релиза и распакуйте папку `ComfyUI-PowerShard` в `ComfyUI/custom_nodes`. При обновлении сохраните прежнюю папку вне `custom_nodes` и замените её новой. Оставьте только одну установленную копию. Перезапустите ComfyUI и обновите страницу браузера.

Используйте Python существующей ComfyUI. PowerShard не устанавливает и не заменяет PyTorch, CUDA, NCCL или пользовательские attention wheels. Сначала проверьте окружение:

```bash
cd ComfyUI-PowerShard
python scripts/diagnose.py
python scripts/check_source_requirements.py
python scripts/dependency_plan.py
```

При необходимости установите только недостающие runtime-зависимости. Инструкции для [x86_64](docs/INSTALL_X86_64.md) и [POWER9 / ppc64le](docs/INSTALL_PPC64LE.md). Веса моделей скачиваются отдельно в стандартные каталоги ComfyUI.

## Быстрый старт

1. Откройте workflow нужного семейства и выберите свои checkpoints, текстовый энкодер и VAE.
2. В `PowerShardConfig` задайте `gpu_ids`, `weight_placement`, `attention_backend` и `sequence_mode`. При `cpu` шарды весов хранятся в RAM, вычисления идут на GPU.
3. Подключите `MODEL` загрузчика PowerShard к родному sampler. Для Wan 2.2 A14B используйте MoE loader и готовый high/low граф.
4. Измените prompt и параметры генерации, затем запустите граф. Имена файлов и наборы GPU в примерах замените своими.

Подробные руководства: [MiniMax H3 / Qwen](README_H3_RU.md), [Wan 2.1 / 2.2](README_WAN22_RU.md), [LTX / Gemma / ускорители / форматы весов](README_LTX_RU.md).

## Проверки и ограничения

Версия **0.5.3** — присланная сборка MiniMax + Wan + LTX. Проверки перед публикацией и точные результаты: [отчёт релиза](reports/release-2026-10-08/verification.json). Исторические результаты H3: [benchmarks](BENCHMARKS.md).

CPU-тесты не подтверждают полную генерацию с настоящими checkpoints на CUDA/NCCL или POWER9. Скорость на GPU, освобождение физической VRAM и ATS требуют проверки на целевой машине. Руководства Wan/LTX содержат native/CUDA probes и ограничения конкретных моделей; общие ограничения — в [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). Кэширование и ускорители могут влиять на качество. Измеренное ускорение и универсальная гарантия отсутствия OOM не заявляются.

Лицензия проекта — [GPL-3.0](LICENSE). Уведомления сторонних компонентов: [NOTICE](NOTICE), [licenses](licenses/). Лицензии моделей действуют отдельно.
