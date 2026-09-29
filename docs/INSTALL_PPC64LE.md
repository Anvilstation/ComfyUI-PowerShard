# IBM AC922 / POWER9 / Ubuntu 20.04

Цель: Python 3.11 + PyTorch 2.12 + CUDA Toolkit 12.4 + V100/sm_70. Ничего не заменяет установленную кастомную сборку. Аппаратная переносимость NOT_RUN_ON_AC922.

## Стек до custom nodes

1. Запустите `scripts/diagnose.py` именно в Python вашей ComfyUI. Строка CUDA Version из nvidia-smi не является версией Toolkit. Сравниваются `torch.version.cuda`, `nvcc --version`, CUDA version.json и драйвер.
2. Используйте Python 3.11 и ваш torch 2.12. Проверяются CUDA availability, sm_70, NCCL и реальные collective operations. Объём каждой V100 определяется, не предполагается 16/32 GiB.
3. У исходного PyTorch v2.12.0 configure gate CUDA >=12.1: CUDA 12.4 проходит **этот конкретный** gate. Это не доказательство всей сборки. Проверены `CMakeLists.txt`, `cmake/public/cuda.cmake`, `cmake/Dependencies.cmake`, `setup.py`. GCC требуется >=11.3. Штатные Python 3.8/GCC 9 Ubuntu 20.04 не отвечают этим требованиям; уже установленная кастомная сборка может использовать другой toolchain.
4. Не делайте вывод о невозможности CUDA12.4/POWER9 из отсутствия официального wheel. Сборка torch в этой сессии не выполнялась. CUDA-facing, NCCL и compiler blockers сверх configure gate пока неизвестны.

Минимальный воспроизводимый тест установочного gate:

```bash
python scripts/check_source_requirements.py
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list()); print(torch.__config__.show())'
python scripts/probe_three.py --gpus 0,1,2 --timeout 120
```

Если исходная сборка необходима, выполняйте её в отдельном окружении, с сохранением install/build logs и git SHA. Для Volta нужен `TORCH_CUDA_ARCH_LIST=7.0`; для ppc64le отключение неподдерживаемого FBGEMM обосновано architecture guard в PyTorch source. Выбор конкретного NCCL/cuDNN и toolchain должен опираться на диагностику уже установленного сервера. Пакет не навязывает новые версии.

## Переносимость зависимостей

| Зависимость | Нужна PowerShard? | Путь на POWER9 |
|---|---|---|
| Ray >=2.48 | Нет | Upstream использует native Ray core/Bazel и vendored deps. Вариант source build существует, но не тестировался; управляющий слой заменён subprocess без изменения H3 математики |
| xfuser >=0.4.4 | Нет | Python orchestration сам по себе не означает переносимость его dependency tree/attention kernels. В PowerShard не устанавливается |
| HF kernels / hf_transfer | Нет | Для kernels 0.17.0 найден ppc64le wheel, но это не доказательство поддержки загружаемых GPU kernels. Здесь kernel launcher/downloads не нужны |
| safetensors 0.6.2 | Да | Найден CPython abi3 manylinux2014 ppc64le wheel. Rust/PyO3 source build остаётся альтернативой; CPU header test не зависит от Rust |
| tokenizers 0.22.2 | Через ComfyUI text encoder | Найдены CPython abi3 и PyPy ppc64le wheels. Rust source build возможен; совместимый transformers range проверяется отдельно, не меняйте установленный transformers молча |
| sentencepiece | Через text stack | C++/CMake source build при отсутствии wheel |
| Comfy Kitchen 0.2.34 | Импортируется ComfyUI | Найден py3-none-any wheel; eager backend и upstream --no-cuda. На ppc64le нужен pure/eager без CUDA/HIP extensions |
| Comfy Aimdo 0.5.3 | Импортируется текущей ComfyUI | Найден pure Python wheel 0.5.3. Dynamic VRAM выключен; native allocator не используется. Native funchook требует отдельной переносимости |
| torchvision / torchaudio | Через ComfyUI | Сохранить установленные builds. Upstream torchaudio2.11 поддерживает torch>=2.11, включая2.12; x86 CPU import/resample прошли. CPython3.11 ppc64le torchaudio wheel не найден; исходная сборка ниже |
| av/FFmpeg | Video output | При отсутствии wheel собрать PyAV против локального FFmpeg. Нужен отдельный audio+video SaveVideo smoke |

В reports/local-portability311.json проверены именно CPython3.11 ABI tags и glibc<=2.31. Safetensors/tokenizers abi3 ppc64le есть; sentencepiece/PyAV требуют source route. Ray wheel отсутствует, xfuser pure wheel есть, но его CUDA transitive deps отдельно проблемны; оба не нужны PowerShard. Это не runtime certification.

## Сборки из исходников без замены torch

Ниже воспроизводимые **рецепты**, их исполнение на ppc64le NOT_RUN. Используйте отдельное venv, где доступен ваш torch, и установленный локальный Rust/C++ toolchain. Не запускайте системные установщики автоматически.

Для safetensors/tokenizers сначала соберите wheels, не устанавливая их:

```bash
python -m pip wheel --no-deps --no-binary=safetensors safetensors==0.6.2 --wheel-dir wheelhouse
python -m pip wheel --no-deps --no-binary=tokenizers tokenizers==0.22.2 --wheel-dir wheelhouse
```

Build isolation может скачать Rust build dependencies; offline сборка требует подготовленного cache/crates. Если установленная версия уже работает, обязательного понижения нет: снимите pin в своей отдельной ветке только после совместимости get_slice/serialization tests и зафиксируйте изменение.

Для eager-only Comfy Kitchen в отдельном checkout SHA из `sources.lock.json`:

```bash
python setup.py --no-cuda --no-hip bdist_wheel
```

Это upstream build option, не фиктивная заглушка. Нужны build dependencies из pyproject, но torch/NVIDIA extras не устанавливаются. Проверьте wheel содержимое и импорт на POWER9. PowerShard attention/INT8 Linear вообще не вызывают Kitchen GPU kernels; Kitchen остаётся зависимостью импортов ComfyUI и других компонентов.

Comfy Aimdo использует Python ctypes wrapper. Native `build-linux-aimdo.sh` выбирает capstone только для ARM, а для прочих архитектур distorm/funchook; нельзя запускать этот script на POWER9 и считать успех гарантированным. При `--disable-dynamic-vram` PowerShard не активирует native allocator. Python-only package можно упаковать из исходников (`python -m pip wheel --no-deps . -w wheelhouse`), затем обязательно проверить native ComfyUI import с отключённым dynamic VRAM. Если другой компонент требует `aimdo.so`, нужен отдельный порт, не пустая shared library.

После каждого кандидата на установку используйте `scripts/dependency_plan.py`; готовые wheels устанавливайте только `--no-deps` после проверки dependencies. Сохраняйте `pip freeze`, torch build config, NCCL maps и wheel hashes.

## Torchaudio с Python3.11/torch2.12

Не нужно подменять torch ради torchaudio2.12: такой wheel отсутствует, а [актуальный upstream](https://github.com/pytorch/audio) прямо объявляет torchaudio2.11 совместимым с torch2.11 и последующими версиями. Проверены HEAD b85c99ccac635a06b1afaf5284bf4c1a00c1f9b5 и tag v2.11.0 34c52a67e8941bbd8e6adaca0eb0b9eabec11d78.

Для ppc64le — отдельный checkout source, сначала изучить setup.py/build options и собрать wheel без dependencies, не менять системную установку:

```bash
python -m pip wheel --no-deps --no-build-isolation /ABS/audio-source --wheel-dir /ABS/wheelhouse
```

Это рецепт, не выполненная здесь POWER сборка. Перед установкой проверить metadata и импорт с вашим torch. Минимальная функциональная проверка после одобренной установки:

```bash
python -c 'import torch,torchaudio; x=torch.ones(1,1600); y=torchaudio.functional.resample(x,16000,24000); assert y.shape==(1,2400) and torch.isfinite(y).all(); print(torch.__version__,torchaudio.__version__)'
```

## NVLink/NUMA и offload

AC922 действительно имеет POWER9↔V100 NVLink2 по [IBM technical overview](https://www.redbooks.ibm.com/abstracts/redp5472.html). Но ваша конкретная GPU↔socket раскладка, PCI BDF, locality выбранных трёх GPU и эффективная полоса здесь неизвестны.

diagnose собирает nvidia-smi topo/P2P/NVLink remote PCI, lscpu/numactl, sysfs numa_node/local_cpulist. Если драйвер не выдаёт поле, записывает ошибку, не подставляет вымышленный node.

numa_policy=none (x86 default): ничего не меняет. auto (AC922 CLI profile): ограничивает CPU affinity worker разрешёнными локальными CPU, allocation first-touch; это не обещание жёсткой memory locality. bind: запускает worker через numactl --physcpubind/--membind, требует разрешений и известной locality. Ошибка bind не обходится.

Сравните три независимых transfer отчёта:

```bash
python scripts/benchmark_transfer.py --gpus 0,1,2 --numa-policy none --output reports/local-transfer-none.json
python scripts/benchmark_transfer.py --gpus 0,1,2 --numa-policy auto --output reports/local-transfer-auto.json
python scripts/benchmark_transfer.py --gpus 0,1,2 --numa-policy bind --output reports/local-transfer-bind.json
```

Каждый измеряет pageable/pinned H2D и D2H, с CUDA synchronization. Затем сравните FSDP CPUOffloadPolicy --pin-memory / --no-pin-memory и prefetch0/1/2 через probe_h3_cuda.py и accept_h3.py. Не предполагайте, что D2H копия полных весов нужна в frozen inference: canonical CPU shards не изменяются.

Суммируйте PSS, а не только RSS, и отдельно CPU shard bytes. Полный checkpoint mmap видим в каждом address space, но loader читает лишь локальные rows; он не является тремя материализованными CPU моделями. Реальные page-cache/locked RAM и пиковая загрузка требуют измерения.

## Запуск

- Ваша текущая ComfyUI. Нет checkout/reset по SHA; допускается текущий API, проверяемый capabilities.
- `--disable-dynamic-vram --use-pytorch-cross-attention`.
- Сначала `probe_three.py`, затем `compare_block.py`, `accept_h3.py`, только после этого UI workflow.
- `native_offload` text encoder не считать проверенным Volta backend. Переносимый исходный профиль использует Qwen CPU FP32; память RAM определите заранее.
- Не устанавливайте `NCCL_P2P_DISABLE=1`/`NCCL_SHM_DISABLE=1` по умолчанию. Изменение transport допускается только как отдельная диагностика с записью в отчёт.

Источники: [PyTorch CUDA gate](https://github.com/pytorch/pytorch/blob/v2.12.0/cmake/public/cuda.cmake), [PyTorch CMake](https://github.com/pytorch/pytorch/blob/v2.12.0/CMakeLists.txt), [safetensors](https://github.com/huggingface/safetensors), [tokenizers](https://github.com/huggingface/tokenizers), [Comfy Kitchen build](https://github.com/Comfy-Org/comfy-kitchen), [Aimdo](https://github.com/Comfy-Org/comfy-aimdo), [CUDA12.4 Volta](https://docs.nvidia.com/cuda/archive/12.4.0/volta-tuning-guide/index.html).
