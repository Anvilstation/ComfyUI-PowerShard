# PowerShard 0.5.1 — разбор регрессии на AC922

Разобран присланный `powershard (1).zip`: 8 session JSON, 7 run JSON, rank logs и `Text Document.txt` с двумя снимками nvidia-smi. Исходные логи не менялись. Машинный разбор, PID, точные байты и SHA256 входных файлов: [reports/performance-0.5.1/uploaded-log-analysis.json](reports/performance-0.5.1/uploaded-log-analysis.json).

**Найден дефект моего планировщика 0.5.0, который воспроизводится на числах из этих логов. Исправления ниже проверены на CPU; новая скорость, качество генерации, CUDA/NCCL и ATS на AC922 здесь не измерены. Вернуть 45 s/iteration по одному CPU-тесту обещать нельзя.**

## 1. Attention и Triton

| Запуск | Запрошено → выбранная policy | Фактические вызовы | Один завершённый H3 forward, rank 0 |
|---|---|---|---:|
| `zvqa3svd`, 4 GPU, token, CPU | vllm_flash_attn → vllm_flash_attn | 2 refiner + 50 DiT vLLM FA, без fallback | 61.56 s |
| `89zaurbs`, 5 GPU, token, CPU | sdpa → sdpa | 2 refiner + 50 DiT SDPA | 45.17 s |
| `hvjcdv7x`, 5 GPU, token, CPU | flash_attn → sdpa | 2 refiner + 50 DiT SDPA | 45.35 s |
| `fkbxjovb`, 5 GPU, Ulysses, CPU | auto → vllm_flash_attn | 2 refiner + 50 DiT vLLM FA, без fallback | 51.99 s |
| `9liuhinq`, Qwen, 5 GPU, GPU weights | auto → vllm_flash_attn | **0 FA; 50 text + 270 vision SDPA** | encode 34.10 s |

У обычного `flash_attn` во всех соответствующих probes — `ModuleNotFoundError: No module named 'flash_attn_2_cuda'`. Наличие Python-пакета не означает, что его CUDA extension доступен этому окружению.

У `vllm_flash_attn` проходят CUDA/numerical probes на V100, включая 56 и 12 heads для пяти карт Ulysses. Загружен `vllm_flash_attn._vllm_fa2_C.abi3.so`, версия `2.7.2.post1+cu124`. H3 действительно вызывает адаптер. Эти логи подтверждают CUDA-вызов и численный smoke, но **не содержат profiler trace, времени конкретного ядра или аппаратных счётчиков Tensor Cores**.

Qwen — отдельный случай: общая policy выбирала vLLM FA, однако все реальные вызовы уходили в SDPA. Причины: FP32/head_dim vision вне области probe и произвольная text mask, которую Flash API не принимает. Mask не удаляется ради ускорения: это изменило бы conditioning. В 0.5.1 UI и `effective_used` показывают фактически использованные пути по группам. У SDPA название backend также не удостоверяет конкретное fused-ядро: это покажет trace.

В production-коде PowerShard нет Triton kernels или `torch.compile`; Triton **не используется непосредственно нодой**. Внешний custom attention provider может иметь собственную реализацию, её нужно смотреть в trace. Для проверки установки/JIT добавлен изолированный Triton smoke; он не объявляется ускорением H3. Upstream Triton сейчас указывает NVIDIA compute capability 8.0+, V100 — 7.0. Это ограничение upstream, а работоспособность вашей custom/старой сборки определяется реальным probe, не названием пакета.

180 W при 99–100% GPU-Util не доказывают поломку FA. Busy time отличается от полезной вычислительной пропускной способности. У повторных cast/reduction, передачи весов из RAM и NCCL есть расходы, которые показания мощности не разделяют. В приведённых запусках SDPA/token оказался быстрее vLLM/Ulysses, но менялся также режим sequence; это не чистый A/B тест провайдеров. `auto` проверяет совместимость, а не выбирает самое быстрое ядро.

## 2. Где действительно находятся веса

Два снимка nvidia-smi связаны с отчётами **по PID**, а не по предположению:

| Снимок | Session / worker PID | CPU weights rank 0 | CUDA tensors после forward | CUDA reserved после forward | Reserved − allocated |
|---|---|---:|---:|---:|---:|
| 14:57:16, 4 карты | `zvqa3svd`, 17661/17663/17665/17668 | 9.386 GiB | 0.274 GiB | 10.943 GiB | 10.669 GiB |
| 15:00:18, 5 карт | `89zaurbs`, 19381/19383/19385/19387/19389 | 7.511 GiB | 0.274 GiB | 10.123 GiB | 9.849 GiB |

В обоих случаях все persistent H3 shards имеют `device=cpu`, pinned=true; `gpu_shard_bytes=0`, `sharded_after_forward=true`. Их суммарный размер по всем ranks **одинаков: 40,312,920,064 bytes**. На пятой карте последний shard чуть меньше из-за row partitioning. Это одна разделённая модель в RAM, не пять полных моделей в VRAM.

Показанные в nvidia-smi десятки GB главным образом соответствуют CUDA allocator cache после временных all-gather/cast/attention/MLP allocations. Во время forward активные GPU tensors тоже действительно занимают память: измеренные пики 8.3–9.3 GiB на rank. После освобождения tensors PyTorch удерживает их storage для повторного использования; nvidia-smi продолжает показывать это как занятую память. Разница reserved−allocated — оценка неактивного cache; при pending stream work нужно учитывать active_bytes, а не только allocated.

На GPU 4 обеих фотографий PID 4675 (`python`, host ComfyUI) занимает около 5950 MiB. Это **другой процесс**, не H3 worker. По снимку нельзя определить, какой host model/cache удерживает эти байты; нода не должна называть их H3 weights. На пяти картах выбран набор `0,1,2,3,5`, GPU 4 в H3 workers не участвует.

В CPU mode GPU всё равно нужен для активного блока, all-gather, activations и workspace. Убрать всю занятую VRAM во время вычислений невозможно. Возвращать allocator cache драйверу на каждом DiT block/step тоже дорого. Cache сохраняется для повторного использования и очищается на end_run/idle либо исчезает при закрытии workers. 0.5.1 уменьшает лишние временные веса/FP32 transfers и показывает память по категориям; **постоянные CPU weights уже были в RAM**.

## 3. Подтверждённая ошибка автоматического MLP

В 0.5.0 я убрал `empty_cache()` после каждого forward, но оставил планирование только по `cudaMemGetInfo.free`. Свободные блоки собственного allocator считались недоступными. Это ошибка: allocator может их повторно использовать.

Пример `89zaurbs` после первого forward: driver free ≈4.04 GiB, reserved≈10.12 GiB, allocated≈0.27 GiB. Старый план вычитает reserve, activation/communication estimates и активную FSDP group: budget становится 0. После вычитания полного MLP output автоматический chunk равен **1 token** вместо тысяч tokens. Для 9307 локальных tokens это 9307 chunks на блок, со многократным масштабированием весов и запуском GEMM.

Арифметический replay после единственного завершённого forward даёт chunk=1 для **всех четырёх** новых H3 session с полным forward, включая Ulysses: там остаток планируемого budget меньше full MLP output. Это проверяется regression tests по исходным байтам. Сам прерванный следующий RPC в старой версии не записывался: его фактическое время/chunk **не измерены**. Дефект подтверждён и согласуется с резким замедлением, но объявлять его единственной причиной всех 100+ секунд было бы неверно.

Другие расходы 0.5.0: Safe QKV/output обмен в FP32; повторное FP32 преобразование и вычисление bounds больших Linear weights; отключение reduced FP16 reductions cuBLAS; в новых H3 runs включён `debug_finite=true`, который сканирует промежуточные tensors. Load times в логах также сильно меняются (≈13–85 s). Полный `sampling_wall_s` прерванного запуска не является ни временем одного forward, ни временем восьми завершённых iterations.

## 4. Что изменено в 0.5.1

1. Планирование учитывает driver-free плюс reusable cache. Active/pending-stream allocations не считаются cache. На входе каждого автоматического MLP читается текущее allocator occupancy, где уже учтены реальные reference/keyframe expansions и materialized weights. Внутри chunk loop нет collectives или CPU tensor reads. Полностью помещающийся MLP больше не округляется вниз с маленьким лишним хвостом. При исчерпанной оценке вместо тысяч single-token GEMMs используется ограниченный minimum chunk до 256 tokens с явной диагностикой; реальный CUDA OOM остаётся ошибкой, гарантия вместимости не заявляется. Manual/off сохраняют своё поведение.
2. Для FP16 Linear scalars/bounds вычисляются один раз на CPU checkpoint shards небольшими tiles; один MAX collective собирает bounds всех Linear. Между блоками сохраняются только числа, не full FP32/dequant weights. Если weight scale=1, GEMM использует исходный materialized FP16 weight view. INT8 storage/ConvRot не заменяются другим quant format; INT8 пока не имеет этого нового checkpoint-bound fast path.
3. По умолчанию Safe sequence использует normalized FP16 QKV/output. Q/K масштабируются по RMS/RoPE bounds; V получает общую степень двойки после MAX по всем ranks **до** cast/exchange. Ulysses восстанавливает масштаб после обратного обмена. Неконечные значения не чинятся clamp. Финальный residual gather остаётся FP32; values >65504 не сужаются. Это вдвое уменьшает logical QKV/output payload относительно FP32; ускорение всей генерации не следует из этого автоматически. `sequence_comm_dtype=fp32` остаётся явным API режимом сравнения; Safe math сохраняет FP32 exchange.
4. В worker/probe разрешены reduced FP16 GEMM reductions только для bounded Safe compute/Qwen; full FP16 accumulation выключена. Absolute-sum bound 16384 с запасом ограничивает частичные суммы ниже FP16 overflow. Флаг и source hash включены в preflight. CPU numerical tests не заменяют проверку этого cuBLAS пути на V100.
5. При разрешённом fallback отсутствующий `flash_attn` теперь пробует доступный `vllm_flash_attn` перед SDPA. Strict mode по-прежнему не меняет явно выбранного провайдера. Fallback не означает, что будет быстрее SDPA; для сравнения нужны одинаковые sequence/shape/seed.
6. В UI добавлен фактический статус по группам и ranks, allocated/cache/CPU shard breakdown. JSON run получает статус COMPLETE/INTERRUPTED_OR_FAILED. Atomic progress сохраняет chosen MLP plan до первого chunk и переживает остановку worker; host включает его в отчёт прерванного RPC.
7. Opt-in profiler записывает реальные CUDA kernel names, SDPA operator, GEMM/attention candidates, all-to-all и NCCL. Дополнительный isolated benchmark проверяет провайдеры на геометрии из этих логов и Triton JIT без установки пакетов. CPU/NOT_RUN не объявляется CUDA PASS.

## 5. Проверка на вашем сервере

Остановите ComfyUI штатным способом, замените папку custom node архивом 0.5.1, полностью перезапустите Python-процесс и обновите страницу браузера. Не оставляйте вторую копию ноды в custom_nodes. Полный restart нужен, чтобы старые импортированные modules и workers не смешивались с новыми файлами. Python, torch/CUDA/NCCL и custom FA не переустанавливаются.

Сначала проверьте маленький настоящий NCCL/FSDP test в используемом окружении ComfyUI. Из папки ноды:

```bash
python scripts/probe_h3_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2,3,5 --sequence-mode token --attention-backend sdpa --weight-placement cpu --output reports/local-5gpu-cpu-token.json
python scripts/probe_h3_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2,3,5 --sequence-mode ulysses --attention-backend sdpa --weight-placement cpu --output reports/local-5gpu-cpu-ulysses.json
```

Microbenchmark реальной attention geometry из присланного forward (total=46535, 56 heads, head_dim=128). Выполняйте при свободных workers; один GPU сначала достаточно для проверки, затем при необходимости повторите на остальных. Return code 2 означает FAIL/NOT_RUN какого-либо выбранного провайдера; все результаты сохраняются по отдельности. Отсутствующий `flash_attn_2_cuda` не мешает отдельному vLLM probe. Если profiler/CUPTI недоступен, сохранённые benchmark/numerical результаты остаются отдельно от `kernels.status=UNAVAILABLE/NO_CUDA_EVENTS`; CUDA illegal access/OOM не скрываются как проблема профилировщика.

```bash
python scripts/probe_accelerators.py --gpus 0 --world 5 --mode token --output reports/local-attention-token.json
python scripts/probe_accelerators.py --gpus 0 --world 5 --mode ulysses --output reports/local-attention-ulysses.json
```

Проверка repeated forward с реальными весами и синтетическим conditioning:

```bash
python scripts/accept_h3.py --comfy /opt/ComfyUI --checkpoint /opt/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors --profile profiles/ac922-v100.json --gpus 0,1,2,3,5 --weight-placement cpu --sequence-mode token --sequence-comm-dtype fp16 --attention-backend sdpa --mlp-mode auto --lifecycle --output reports/local-real-h3-token
```

По умолчанию этот acceptance test малый; добавленные `--frames`, `--latent-height`, `--latent-width`, `--audio-tokens`, `--text-tokens` позволяют проверить длинный workload. Размеры latent не являются пиксельным разрешением. Качество и исходный FL2VA payload проверяйте в своём workflow, поскольку синтетический test не заменяет генерацию.

Для замера верните **тот же** workflow/seed/CFG/resolution/frames/conditioning. Начните с 5 GPU `0,1,2,3,5`, CPU weights, token, SDPA, FP16 Safe on, `debug_finite=false`, Spectrum off, prefetch=0. Это рекомендация по вашим завершённым timing samples, не обещание превосходства SDPA на других shapes. Выполните несколько последовательных шагов; смотрите отдельные `forward_s` и выбранные MLP chunks. Затем сравните vLLM FA, меняя только attention, и отдельно Ulysses. Debug finite включайте для диагностики NaN или отдельного correctness run.

Для одного отдельного profiling run перед запуском ComfyUI штатной командой:

```bash
export POWERSHARD_PROFILE=1
export POWERSHARD_PROFILE_FORWARDS=2
```

Профилирование добавляет overhead и может создавать большие trace files; его время не используйте как обычный benchmark. В `output/powershard` будут rank logs, run/session JSON, `*-progress.json` и `*.trace.json`. `kernels.attention_operators` показывает SDPA dispatch; `attention_kernel_names`/`gemm_kernel_names` — наблюдавшиеся CUDA names. `triton_kernel_names` — имена, содержащие triton, а отсутствие такого имени само по себе не доказывает отсутствие внешних сгенерированных kernels. Для процентной загрузки Tensor Cores нужны аппаратные counters, например отдельный Nsight Compute run; wattage их не заменяет. После profiling уберите две environment variables перед обычным benchmark.

Для CPU/GPU/ATS weight audit остаётся `tests/audit_memory.py`. В архиве есть старые Qwen runs с `weight_placement=ats`, `cpu_offload=true`, CPU shards и timeout_s — это прежний CPU fallback, **не доказательство работы нового managed ATS**. 0.5.x запрещает такое переименование режима; нужны managed pointer evidence и актуальный source hash.

## 6. Доказательства и оставшиеся проверки

Автоматические результаты текущей ревизии: [reports/performance-0.5.1/verification.json](reports/performance-0.5.1/verification.json), pytest XML/log и native CPU sampler contract рядом. Проверены large-value Safe compute, normalized half exchange/final FP32 residual на 3/4/5/6 ranks с emulated transport, равенство native outputs, cache-budget replay, interrupted RPC progress, strict/fallback semantics и синтаксис UI. Настоящие Gloo tests честно пропущены из-за EPERM на TCP transport; здесь нет V100/POWER9 для CUDA/NCCL tests.

Следующие приоритеты производительности после исправления: profile actual workload; A/B prefetch 0/1 с текущим memory budget; проверить H2D/NUMA и NCCL overlap; сохранить точную Qwen mask semantics при специальном varlen/causal path; отдельно исследовать bounded INT8 dequant fast path. Эти изменения в 0.5.1 не объявляются реализованными или ускоренными.

## Первичные источники

- [PyTorch 2.12 CUDA semantics: allocator, allocated/reserved, FP16 reduction и V100 timings](https://docs.pytorch.org/docs/2.12/notes/cuda.html).
- [FSDP2 CPUOffloadPolicy: CPU shards → H2D → all-gather, reshard lifetime](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html).
- [Triton upstream supported hardware](https://github.com/triton-lang/triton#compatibility).
- [Flash Attention upstream requirements](https://github.com/Dao-AILab/flash-attention). Это не спецификация вашей custom sm_70 сборки; её реальные probe/call данные взяты из присланных логов.
