# Linux x86_64 / CUDA / V100

Код тот же, что на POWER9. Профиль `profiles/x86-v100.json` отличается названием назначения, не семантикой шардирования. Реальные модели и аппаратура NOT_RUN.

1. Сохраните диагностику уже установленного torch/ComfyUI. Не обновляйте драйвер/CUDA/NCCL автоматически.
2. Требуется torch 2.12.x с CUDA и NCCL и скомпилированным sm_70 для V100. Более новый CUDA wheel без sm_70 не подходит даже при успешном `import torch`. Пакет проверяет arch list и запускает вычислительный probe.
3. Используйте текущую ComfyUI с Python3.11. SHA/version не ограничивают допуск; source_guard проверяет методы/сигнатуры. Пользовательские изменения не сбрасываются.
4. Поместите PowerShard в custom_nodes, выполните dependency dry-run и затем установку только необходимого safetensors.
5. Запустите `python scripts/probe_three.py --gpus 0,1,2`. Индексы можно заменить полными GPU UUID. Все три rank обязаны завершить compute/collectives/FP16/INT8 smoke.
6. Выполните probe_h3_cuda.py с --comfy, затем тот же probe с --cpu-offload. После PASS — FP16 Safe workflow, INT8, offload и отдельно sequence. Сохраняйте environment, revisions, output, seed, steps и rank logs.

Более новые GPU: пакет проверяет capability, но не активирует BF16/FP8/NVFP4, Triton или fused attention. Они потребуют отдельного backend/численной проверки; запуск нашего FP16 backend не является проверкой этих ускорителей.

На машине с существующим рабочим torch используйте этот interpreter; приведённый в отчётах тестовый `2.12.0+cpu` предназначался только для CPU CI текущей сессии, **не** для сервера с V100.
