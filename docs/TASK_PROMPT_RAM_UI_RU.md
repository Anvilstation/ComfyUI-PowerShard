> Историческая документация 0.5.0rc1. Профиль `ram_min` и описанные здесь UI/API относятся к старой версии. Текущие настройки и миграция: [README_RU.md](../README_RU.md); изменения: [PERFORMANCE_FIX_0_5_3_RU.md](../PERFORMANCE_FIX_0_5_3_RU.md).

# Промпт задачи: RAM-профиль, ATS, интерфейс и оптимизации

Доработай существующий ComfyUI-PowerShard на основе присланного кода и экспериментальных исправлений аудита 2026-09-28. Работай в отдельной ветке существующего Git-репозитория. Сохрани исходники, class IDs нод, старые workflows, выбранные CUDA GPU, FP16-safe, INT8 ConvRot, FSDP2, Qwen CLIP, audio/video conditioning и выбранный attention provider. Не изменяй production ComfyUI, torch/CUDA/NCCL/драйвер или пользовательский wheel.

1. Реализуй явно выбираемый профиль `ram_min` («Минимум VRAM / веса в RAM»). Постоянные локальные шарды H3/Qwen хранятся в CPU RAM, активные параметры временно собираются на GPU и снова шардируются. Не создавать N полных CPU/GPU моделей. Профиль должен отключать prefetch и prepared-weight cache, уменьшать FSDP-группы, ограничивать MLP/Linear temporary workspace и исключать постоянный CUDA-кэш conditioning. Сохранить no-backward inference hooks и реальное шардирование. Настройки профиля должны одинаково доходить до всех workers и входить в fingerprint.

2. Для SDPA/fallback ограничить временную attention-матрицу обработкой частей Q/heads без изменения полного attention. Сохранить заданные scale, masks, GQA, causal alignment, окно, ALiBi и dropout; не применять clamp/nan_to_num. Optional custom kernels не подменять молча. Ограничение workspace — цель планирования, не гарантия отсутствия OOM и не запрет по прогнозу. Не увеличивать reserve искусственно до фактического использования всей VRAM.

3. Исследуй настоящий ATS/Unified Memory для POWER9/V100 по официальной CUDA 12.4 и установленной PyTorch 2.12 реализации. Раздели UVA, ATS, managed/mapped allocations, NVLink и CPUOffloadPolicy. Не выдавай обычный DMA offload за ATS и не обещай «OOM невозможен». Дай технический вывод и воспроизводимую изолированную диагностику. Не внедряй непроверенный глобальный allocator в ComfyUI.

4. Приведи интерфейс в порядок: русские названия, описание назначения каждой ноды, человекочитаемые подписи и подсказки параметров, панель настроек с основными/расширенными полями и явным описанием компромисса RAM/VRAM/скорость. Сохрани порядок прежних сериализованных widgets; новые поля только в конце. Не переименовывай API keys и типы сокетов. Импорт без optional kernels/GPU остаётся безопасным.

5. Подготовь практические рекомендации по экономии памяти, скорости и качеству. Отдели точные оптимизации от приближённых Spectrum/Sage/quantization и от настроек видеоencoder. Не обещай ускорение от большего числа GPU. Подготовь INT8/FP16 и SDPA/custom-provider workflows, сохраняя audio и PNG после VAE.

6. Выполни исходные тесты и регрессии аудита. Добавь проверки effective RAM policy, передачи в workers, FSDP wrapping ownership/hooks, bounded cache, tiled GEMM/attention vs reference, metadata INT8, повторов, clone isolation и старых UI graphs. CPU mocks не считать CUDA доказательством. Аппаратные тесты без AC922/доступных GPU пометить NOT_RUN и дать точные команды с действующим Python ComfyUI.

Результат: исходники в существующем проекте, этот промпт, workflows, документация, runnable hardware smoke/benchmark, raw результаты доступных тестов и пакет для отдельного проверочного запуска. В отчёте явно указать, какие расходы VRAM остаются обязательными, что реализовано, что проверено и что ещё требует AC922.
