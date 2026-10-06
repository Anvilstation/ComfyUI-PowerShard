> Историческая документация до 0.5.0. Старые UI/CLI аргументы и результаты не относятся к текущей ревизии. Актуальные команды: [README_RU.md](../README_RU.md), результаты: [AUDIT_REVIEW_2026-10-04.md](../AUDIT_REVIEW_2026-10-04.md).

# Выбор GPU и attention — 0.3.0

Это изменение существующего PowerShard 0.2.0 (`9694964e68d58f7aa5228817dc65ecff9e7cda76`), не новый backend. Более поздние серверные изменения, включая предполагаемые LoRA descriptors, в доступном архиве отсутствовали. Их нельзя считать проверенными или автоматически перенесёнными.

## Устройства

`gpu_ids`: `0`, `0,1`, `1,3,5`, `5,2,0`, `all`, полные UUID. Индексы относятся к CUDA GPU, **видимым исходному ComfyUI process**, а не к физической нумерации nvidia-smi. При `CUDA_VISIBLE_DEVICES=5,2,0` выбор `1,0` означает видимые карты 1 и 0, то есть соответствующие UUID физических 2 и 5. Авторитетен CUDA inventory, не предположение о PCI-порядке.

`all` выбирает весь доступный набор, включая шесть и больше карт. Порядок сохраняется. Дубликат предупреждает; пустой/несуществующий ID — ошибка. Список не расширяется. Workers видят только выбранные UUID: rank и локальный CUDA index — 0..N−1. Диагностика хранит исходный индекс, UUID, rank, worker index, name/VRAM.

Config node содержит кнопку с inventory, чекбоксами и выбранным количеством; строка `gpu_ids` остаётся источником истины для JSON/API. Новые widgets добавлены после всех старых, class IDs неизменны. В отсутствие новых полей используется прежний `math`/exact-chunked режим. `allow_fallback=true` по умолчанию.

N используется в spawn, NCCL preflight, DeviceMesh, shard boundaries, ответах, метриках и shutdown. При N=1 используется та же FSDP2 infrastructure, но `inter_gpu_sharding=false`: распределения между GPU нет. NCCL/FSDP2 runtime нужны и для этого пути. При N>1 остаётся block/root FULL SHARD с reshard после forward, без полных постоянных копий. Колонки ConvRot-групп не режутся: вес и row scales делятся по одинаковым выходным строкам. Проверяются также неделимые размеры и пустые хвостовые shards.

Sequence backend делит tokens, не heads: `heads % N` не требуется. Если короткая последовательность даёт пустой rank, все ranks этого forward используют FSDP-only, без изменения выбранных GPU. Метрики показывают фактический режим. Ускорение этой схемы не измерено.

## Attention

| Выбор | Реальный путь |
|---|---|
| auto | Общий для выбранных GPU прошедший probe: SDPA, затем custom vLLM, upstream Flash, math. Sage не выбирается. Не обещание fastest. |
| sdpa | `torch.nn.functional.scaled_dot_product_attention`, automatic dispatch, без FLASH-only context |
| flash_attn | Lazy import установленного `flash_attn`, dense либо varlen entrypoint |
| vllm_flash_attn | Внутренний `VLLMFlashAdapter` над установленным kernels-пакетом; весь vLLM server не нужен |
| sageattention | Explicit opt-in квантованного attention, отдельный численный допуск |
| math | Полный tiled online-softmax; FP32 accumulation. При FP16-safe patch GEMM использует прежнее bounded half scaling; независимый тестовый oracle — FP32. |

Вызовы установлены на **instances настоящих H3 Attention** внутри каждого worker: main DiT и token refiner. Native H3 packing, modulation, audio/video/ref conditioning не переписаны. FSDP вызывает root/block `__call__`, hooks не обходятся. Счётчики `dit:<provider>` / `token_refiner:<provider>` показывают реальный путь, включая semantic fallback. Нулевые счётчики не являются включённым FlashAttention.

Внутренний layout всегда **BLHD**. Q=[B,Lq,Hq,D], K=[B,Lk,Hkv,D], V=[B,Lk,Hkv,Dv]. GQA требует Hq кратное Hkv; batch и K/V lengths совпадают. Bool mask True разрешает связь; additive mask прибавляется к scores. Scale применяется до softmax. Causal по умолчанию upper-left, как SDPA; bottom-right задаётся явно. Window и ALiBi используют тот же positional offset. Fully masked rows возвращают ноль, NaN данных не маскируется.

Flash adapter формирует разные `cu_seqlens_q/k`, max lengths и формы Q/K/V. Для packed API задаются CPU `lengths_q/k`; отсутствует GPU→CPU чтение lengths на каждом блоке. Поддерживаются положительные неравные длины batch. Unknown kwargs дают TypeError. Signature исследуется один раз, `**kwargs` не считается доказательством поддержки. Необязательный аргумент можно не передать только когда его отсутствие эквивалентно default.

`return_attn_probs=True` означает `(output, FP32 LSE, probabilities после dropout)`; непрозрачный Flash `S_dmask` не выдаётся за probabilities. Этот запрос переходит в math или завершается UnsupportedAttention при запрете fallback. Большой diagnostic probabilities tensor требует B×H×Lq×Lk памяти; обычный math его не создаёт. Dropout не обнуляется.

### Область CUDA probe и fallback

Перед NCCL workers запускаются отдельные короткие процессы на каждом выбранном UUID: импорт, origin/distribution metadata/signature, реальный CUDA forward, два нестандартных scale, cross lengths, B=2, non-contiguous Q, square causal и FP32 reference. После этого выбирается **единая policy** на всех ranks. Сбой/timeout child не оставляет испорченный CUDA context в ComfyUI.

Production custom-provider policy пока сертифицирует FP16/head_dim текущего H3, обычный equal-head attention, без dropout/window/ALiBi. Эти дополнительные возможности реализованы в Python adapter и покрыты CPU contract tests, но для произвольной CUDA-сборки ещё не сертифицированы; production dispatcher переводит их в SDPA/math с причиной. Rectangular causal также идёт в математически согласованный fallback, а не доверяет версии Flash. Это осознанное ограничение, не молчаливая потеря аргументов. `allow_fallback=false` вместо перехода выдаёт конкретную ошибку.

Signature/probe не гарантируют работу всех длин: настоящий OOM/illegal access/assert/неверный output contract **не ловятся** для повторения блока. Session завершается целиком; новый запуск создаёт чистые workers. Fallback ловит только собственный `UnsupportedAttention`, выбрасываемый до kernel. Unknown hardware/version не является причиной отказа.

Идентичность provider, policy, выбранных UUID и patch входит в session fingerprint. Изменение Config/patch создаёт новую session; старые workers освобождаются. Файловый stamp provider инвалидирует loader cache; содержимое `.so`, заменённое вручную под старым импортированным Python-модулем, требует перезапуска ComfyUI. Никаких больших tensors в capability cache нет.

## FP16 Safe и INT8

Condition/residual/native safety islands остаются FP32, Linear/MLP — прежние scaled FP16 GEMM. Перед fused attention Q/K масштабируются степенями двух по **аналитической границе RMSNorm и RoPE**, вычисленной один раз из небольших norm vectors checkpoint. Scale компенсируется `softmax_scale *= sq*sk`. V масштабируется GPU tensor `sv` и после attention восстанавливается `out.float()*sv`. Никакой повторной компенсации, clamp или nan_to_num. Это сохраняет FP32 residual без обязательного FP32 fused kernel и без `.max().item()` в блоках. Конечность проверяется прежним deferred tracker на RPC boundary.

INT8 parameter storage, row scales и ConvRot metadata не меняются. Attention adapter получает уже вычисленные Q/K/V и не деквантует генератор. FSDP/offload/prefetch остаются независимыми настройками. Half rounding/underflow остаются; качество реального H3 требует аппаратного сравнения.

## Переезд с глобального shim

Прочитан архив `flash_attn_shim(1).zip`, SHA256 `0791ba1ae57fb818fbe4fecd296fee12f71de4c933db1a0a62e1852b68e33076`. Он содержит только Python и pyc, не пользовательский wheel/kernel. Подтверждены потеря scale/options, ошибочный reshape K/V, неполный return contract и sys.path shadowing. Исходники shim не устанавливаются и не импортируются; `.pyc` не используются.

1. Остановите ComfyUI; сохраните резервную копию **точно установленной** папки shim и локальных изменений PowerShard.
2. Уберите эту отдельную custom-node папку из каталога автозагрузки (например перенесите рядом с ComfyUI, не внутрь custom_nodes). PowerShard ничего не удаляет.
3. Если flash_attn.py вручную копировался в site-packages, сначала определите origin через диагностику; сохраните копию и восстановите нужный пакет своим штатным способом. Не удаляйте файлы по предположению.
4. Перезапустите процесс: удаление папки не очищает уже импортированный глобальный модуль. Выберите `vllm_flash_attn` в Config.

Диагностика показывает legacy_shim по path/version и не выдаёт `2.7.2+shim` за версию CUDA distribution. Проект не меняет sys.path для shadowing flash_attn, sys.modules, torch SDPA или глобальные ComfyUI classes.

Пользовательский wheel уже установлен? **Ничего переустанавливать не нужно.** Если требуется явная первичная установка, используйте Python активного ComfyUI окружения, сначала просмотрите:

```bash
python -m pip install --dry-run --no-deps /ABS/vllm_flash_attn-2.7.2.post1+cu124-cp311-cp311-linux_ppc64le.whl
# Только после проверки плана, если пакет ещё не установлен:
python -m pip install --no-deps /ABS/vllm_flash_attn-2.7.2.post1+cu124-cp311-cp311-linux_ppc64le.whl
```

Никаких pip -U, установки vLLM server, смены torch/CUDA/NCCL или редактирования ComfyUI core. На ppc64le отсутствие upstream wheel не является запретом пользовательской сборки.

## Точный первый запуск

Из `ComfyUI/custom_nodes/ComfyUI-PowerShard`, с Python активного окружения:

```bash
python scripts/diagnose.py --comfy ../.. --output reports/server-environment.json
python scripts/diagnose_attention.py --gpus all --backend vllm_flash_attn --no-allow-fallback --checkpoint /ABS/minimax_h3_fl2va_pruned_int8_convrot.safetensors --output reports/server-custom-attention.json
python scripts/probe_devices.py --gpus all --attention-backend sdpa --timeout 180
python scripts/probe_h3_cuda.py --comfy ../.. --gpus all
python scripts/probe_h3_cuda.py --comfy ../.. --gpus all --cpu-offload
python scripts/benchmark_transfer.py --gpus all --numa-policy auto
python scripts/benchmark_attention.py --gpus all --backend vllm_flash_attn
bash scripts/launch_comfy.sh "$(command -v python)" "$(realpath ../..)"
```

`--no-allow-fallback` — диагностика custom provider, не обычный режим. При его FAIL основной workflow может использовать SDPA/math. На сервере с одной GPU укажите `--gpus 0`; сокращение `--single-rank` само набор не урезает. Старое имя `probe_three.py` оставлено CLI alias по совместимости и больше не ограничивает N.

Откройте `workflows/fl2va_all_sdpa.ui.json` (INT8 FL2VA, все видимые GPU, FP16 patch, обычный sampler, AV VAE) или `fl2va_all_vllm_offload.ui.json`. Выберите реальные локальные checkpoints/энкодеры/VAE и prompt. В этих новых графах SaveImage сохраняет PNG непосредственно из video VAE **до** CreateVideo/SaveVideo. У `fl2va_subset_math.ui.json` пример `5,2,0`: измените его, если такие индексы недоступны. Это пример, не обязательный набор.

Сравнение качества: одинаковые checkpoint revision, LoRA (в этой доступной базе не поддерживается), seed, sampler, steps, resolution/frames/offload. Сравнивайте PNG и затем кодированное видео отдельно. Ошибки shim не доказывают причину розовых артефактов.

## Источники API

Проверены [SDPA PyTorch 2.12](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html), [FlashAttention interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py), [SageAttention core](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py). Это справка по контрактам, а не доказательство свойств пользовательского wheel. Авторитетны установленная сигнатура и собственный CUDA probe. ComfyUI source проверен на `7a0b5eede3f9721c8faab290689893f36edc6d66`; версия не gate.
