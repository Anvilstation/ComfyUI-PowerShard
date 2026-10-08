# PowerShard 0.5.3 — MiniMax H3 + Wan / LTX

## Русский

Публичный выпуск ComfyUI-PowerShard с поддержкой MiniMax H3 и Wan 2.1 / 2.2 в обычных графах ComfyUI. В сборку также входят LTX-Video / LTX-AV, распределённые umT5 / Gemma и дополнительные ускорители.

- FSDP2 шардирование весов и token / Ulysses sequence parallel на выбранном наборе CUDA GPU.
- MiniMax H3 FL2VA / Ref2VA, FP16 Safe, INT8 ConvRot и Qwen3-VL conditioning.
- Wan T2V / I2V, A14B high/low MoE, TI2V-5B, LoRA и специализированные варианты.
- GPU / CPU размещение весов, RAM parking между задачами, экспериментальный ATS.
- Готовые workflows и описание установки на русском и английском.

Установка: распакуйте `ComfyUI-PowerShard-0.5.3.zip` в `ComfyUI/custom_nodes`, заменив прежнюю папку, затем перезапустите ComfyUI. Веса моделей скачиваются отдельно; существующие torch/CUDA/NCCL и attention wheels не заменяются. Подробности: [Русский](README_RU.md), [English](README.md).

Проверки перед публикацией: **276 passed, 153 skipped, 0 failed** на Windows x86_64 / Python 3.11.15 / torch 2.12.0+cpu, включая реальные Gloo/Ulysses на 3–6 процессах. Проверены синтаксис Python/JavaScript и регистрация 32 нод. Native ComfyUI, CUDA/NCCL, POWER9, реальные checkpoints и GPU-производительность здесь не проверялись. [Отчёт](reports/release-2026-10-08/verification.json).

Основа выпуска — присланный ZIP со сборкой 0.5.3. При подготовке добавлены двуязычные README и метаданные публикации; подробный H3 README сохранён отдельно. Исправлен расчёт метрики ошибки в тесте Ulysses для пустого шарда, без изменения кода генерации. Исходный архив и SHA-256 отмечены в отчёте; ZIP релиза соответствует опубликованному commit и содержит эти изменения.

## English

Public ComfyUI-PowerShard release supporting MiniMax H3 and Wan 2.1 / 2.2 in regular ComfyUI workflows. The build also includes LTX-Video / LTX-AV, distributed umT5 / Gemma encoders, and additional accelerators.

- FSDP2 weight sharding and token / Ulysses sequence parallelism across selected CUDA GPUs.
- MiniMax H3 FL2VA / Ref2VA, FP16 Safe, INT8 ConvRot, and Qwen3-VL conditioning.
- Wan T2V / I2V, A14B high/low MoE, TI2V-5B, LoRA, and specialized variants.
- GPU / CPU weight placement, RAM parking between runs, and experimental ATS.
- Example workflows and installation descriptions in Russian and English.

Installation: extract `ComfyUI-PowerShard-0.5.3.zip` into `ComfyUI/custom_nodes`, replacing the previous folder, then restart ComfyUI. Download model weights separately. Existing torch/CUDA/NCCL and attention wheels are preserved. Details: [English](README.md), [Русский](README_RU.md).

Publication checks: **276 passed, 153 skipped, 0 failed** on Windows x86_64 / Python 3.11.15 / torch 2.12.0+cpu, including real Gloo/Ulysses with 3–6 processes. Python/JavaScript syntax and registration of 32 nodes were checked. Native ComfyUI, CUDA/NCCL, POWER9, real checkpoints, and GPU performance were not tested here. [Report](reports/release-2026-10-08/verification.json).

Based on the supplied 0.5.3 ZIP. Publication changes add bilingual READMEs and metadata while retaining the detailed H3 guide. The Ulysses test error metric now handles an empty shard; generation code is unchanged. The report records the source archive hash. The release ZIP matches the published commit and includes these publication changes.

Licensed under GPL-3.0; model licenses apply separately. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
