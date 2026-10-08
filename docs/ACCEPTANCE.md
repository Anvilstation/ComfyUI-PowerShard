> Историческая документация до 0.5.0. Старые UI/CLI аргументы и результаты не относятся к текущей ревизии. Актуальные команды: [README_RU.md](../README_RU.md), результаты: [AUDIT_REVIEW_2026-10-04.md](../AUDIT_REVIEW_2026-10-04.md).

# Приёмка на целевом сервере

Не считать кодовое наличие режима его аппаратной приёмкой. Для каждого прогона сохраните дату, architecture, driver/toolkit/torch/NCCL, UUID/topology, checkpoint revision/SHA256, shape, frame count, seed, steps, dtype/backend и output.

## Доступные автоматические проверки

0.2.0: FP16 Patcher включён во всех обновлённых workflows и по умолчанию в accept_h3.py. ComfyUI SHA/version gate отсутствует. Используйте Python3.11 с вашим CUDA torch, не тестовое CPU venv.

```bash
python -m pytest tests -q --comfy ../..
python scripts/probe_three.py --gpus 0,1,2
python scripts/probe_h3_cuda.py --comfy ../.. --single-rank --output reports/local-single-cuda.json
python scripts/probe_h3_cuda.py --comfy ../.. --output reports/local-three-cuda.json
python scripts/probe_h3_cuda.py --comfy ../.. --cpu-offload --output reports/local-three-offload.json
python scripts/probe_h3_cuda.py --comfy ../.. --cpu-offload --no-pin-memory --output reports/local-three-pageable.json
python scripts/probe_h3_cuda.py --comfy ../.. --cpu-offload --prefetch-blocks 1 --output reports/local-prefetch1.json
python scripts/probe_h3_cuda.py --comfy ../.. --cpu-offload --prefetch-blocks 2 --output reports/local-prefetch2.json
```

--single-rank — только диагностика, не трёх-GPU приёмка. Production session всегда три ranks. Native CUDA probe проверяет tiny H3 FP16/ConvRot256, condition input100000, AV forward, пять повторов, реконструкцию только малых параметров и hooks/reshard.

CPU tests с --comfy используют настоящий SamplerCustomAdvanced и отдельный CPU subprocess; без --comfy native tests отмечаются skipped/NOT_RUN. Это не доказательство CUDA.

| Этап | Команда из папки PowerShard | Критерий |
|---|---|---|
| A: импорт/CPU math | `python -m pytest tests -q` | finite attention/ConvRot, exact partition, metadata, early rejection |
| A: native H3 CPU | `python scripts/smoke_native_cpu.py --comfy ../..` | tiny native H3 video/audio/masks/refs и real meta shape match |
| A: Comfy interface | `python scripts/test_comfy_contract.py --comfy ../..` | native Euler sampler, clone, повтор seed через TEST_LOCAL_CONTRACT_ONLY |
| B/C: реальные 3 GPU | `python scripts/probe_three.py --gpus 0,1,2` | все collectives, FP16/INT8 FSDP reconstruction, local intervals, пять forwards без backward |
| D: H3 FP16 | `python scripts/accept_h3.py --comfy ../.. --checkpoint /ABS/H3_FL2VA_BF16.safetensors --lifecycle` | обе ветки finite, повтор forward, отмена и чистая следующая сессия |
| E: H3 INT8 | та же команда с `--precision int8_fp16 --checkpoint /ABS/H3_FL2VA_INT8.safetensors` | I8 local bytes, finite обе ветки, нет постоянных expanded weights |
| F: SP | та же FP16 команда с `--backend fsdp2_sequence` | три ranks, совпадение с FSDP-only, разные локальные token ranges |
| G: граф | `python scripts/run_api_workflow.py workflows/fl2va_fp16.api.json --allow-unverified --submit` | реальный prompt, encoder, denoising, оба VAE и сохранённый AV |

Замените `/ABS/...` фактическими путями, а не создавайте пустые файлы. На этапе D синтетическое conditioning используется намеренно, чтобы отделить генератор от encoder. Оно не подтверждает качество естественного текста.

## Численная проверка

- Tiny FP16 vs reference: `atol=0.002`, `rtol=0.003` для отличий half accumulation; CPU attention FP32 отдельно `rtol=2e-5`, `atol=2e-6`.
- ConvRot dequant vs Comfy Kitchen eager: точное совпадение CPU FP16 результата на тестовом input. Это не качество реального INT8 H3.
- Реальный блок: `compare_block.py`, BF16 storage → FP32 reference против FP16+FP32 islands, default `atol=0.03,rtol=0.03` для синтетических входов. Допуск диагностический, его нельзя автоматически переносить на качество ролика.
- Реальный FSDP-only/SP: сохраните одинаковые synthetic inputs и выходы; сравните max/RMS ошибки и отдельно итоговые video/audio latents. Постоянный seed сам по себе не гарантирует одинаковые inputs, если менялась обработка conditioning.
- Проверьте FP32 reference на нескольких блоках, включая первые/последние, затем реальные intermediate activations. Большая модель целиком не требуется на одной GPU.

## Проверки lifecycle, требующие целевого сервера

Дополнительная регрессия реального condition_proj:

```bash
python scripts/accept_h3.py --comfy ../.. --checkpoint /ABS/H3_FL2VA_INT8.safetensors --precision int8_fp16 --text-magnitude 100000 --debug-finite --output reports/local-condition-stress
python scripts/accept_h3.py --comfy ../.. --checkpoint /ABS/H3_FL2VA_INT8.safetensors --precision int8_fp16 --cpu-offload --prefetch-blocks 1 --lifecycle --output reports/local-offload
```

Не выключайте finite checks ради PASS. Проверяйте три patch fingerprints, condition_proj/residual FP32, I8 shard storage. После этого нужен настоящий Qwen conditioning и весь workflow; синтетический input не подтверждает качество.

После удаления/disabled patch node исходный MODEL должен иметь исходную policy; disabled на V100 может закономерно вернуть старый overflow. Проверяйте source/clone и чистую новую session, не глобальное выключение чужих patches.

`accept_h3.py --lifecycle` проверяет закрытие, отмену ожидаемого RPC и следующий запуск. Дополнительно выполнить и зафиксировать:

1. Три задания с seed 44/45/44 и изменением prompt. Первое/третье сравнить в заданных допусках. CUDA synchronize обязателен при тайминге.
2. Два разных checkpoints последовательно. Сессии не должны сохранять старые GPU tensors; PID и allocator memory после release сверяются с inventory.
3. Во время своей H3 сессии завершить **только один PID из PowerShard logs**. Убедиться, что host сообщает rank error, закрывает остальных двух и следующая генерация стартует. Не использовать pkill/killall.
4. Реальный OOM провоцировать только отдельным тестовым workflow на своих GPU, без чужих задач; сохранить ошибку и подтвердить следующий небольшой workflow. В пакете нет безусловного memory allocation bomb.
5. Убедиться в отсутствии orphan PIDs после штатного закрытия ComfyUI, interrupt и timeout. Для `kill -9` host предусмотрен worker watchdog по parent PID; подтвердить его работу на целевой ОС, включая прерывание CUDA collective.
6. Сопоставить per-rank `shards`, `local_bytes` и `sharded_after_forward` со средствами profiler. Активные DTensor полные значения разрешены только у текущего блока. Reconstruction всей H3 на GPU запрещён; tiny probe делает это только для маленькой модели.

## Память и измерения

Сохранять per-GPU NVML used/free и per-process allocated/reserved/peaks. NCCL allocations не всегда видны `torch.cuda.memory_allocated`, поэтому дополнительно нужна GPU-wide память. PyTorch reset_peak/synchronize есть в worker.

`POWERSHARD_PROFILE=1` экспортирует Chrome trace первых двух RPC каждого rank, включая CUDA/NCCL events. Учтите profiler overhead; latency сравнивайте отдельным прогоном без trace. Trace не заменяет end-to-end benchmark.

Для измерения стадий включите профилировщик **до запуска тестовой ComfyUI** (путь заменить на свой):

```bash
POWERSHARD_BENCHMARK_DIR=/ABS/reports/e2e python main.py --disable-dynamic-vram --use-pytorch-cross-attention
```

Используется audited `execution.py`. Записываются queue-free latency, реально исполненные nodes, cache hits, host CUDA allocated/reserved/peak, sampled GPU-wide memory. Отдельные стадии: чтение encoder checkpoint, CLIPTextEncode, sampler, video VAE, audio VAE, сохранение файла. Синхронные native nodes измеряются целиком; deferred async/subgraphs явно исключаются из stage summary. Включение profiler не создаёт CUDA contexts/workers при импорте custom nodes.

Загрузка H3 отложена до sampler, VAE device load может происходить в decode: node timings включают эти операции и так названы в отчёте. Worker reports отдельно дают generator load, input serialization/transfer, forward и IPC. Вложенные времена нельзя складывать. Отчёты разных источников связываются по времени и одному изолированному заданию; за один benchmark не запускайте конкурентную очередь.

```bash
python scripts/benchmark_summary.py /ABS/ComfyUI/output/powershard/powershard-SESSION.json --prompts /ABS/reports/e2e/prompt-*.json > reports/local-benchmark-summary.json
```

Throughput считается только для успешных заданий с фактически исполненным sampler: jobs/execution-second и jobs/observation-second в пределах одного host process. Полностью cached graph не считается новой генерацией. Для cold/warm runs различайте release_after_sampling true/false, модельные загрузки и graph cache. Точность CUDA синхронизации и всей instrumentation на сервере пока NOT_RUN; два CPU теста проверили только учёт границ/кэша. `nvidia-smi utilization=100%` не является доказательством полезного SP.

Для сравнения FSDP-only и SP задайте разные output directories (иначе второй прогон заменит latents первого):

```bash
python scripts/accept_h3.py --comfy ../.. --checkpoint /ABS/H3_FL2VA_BF16.safetensors --output reports/local-fsdp
python scripts/accept_h3.py --comfy ../.. --checkpoint /ABS/H3_FL2VA_BF16.safetensors --backend fsdp2_sequence --output reports/local-sp
python scripts/compare_runs.py reports/local-fsdp reports/local-sp
```

Все три команды используют один checkpoint и одинаковые synthetic inputs. Перед сравнением локальных файлов проверьте полный SHA256 отдельно. Сравнение одного forward не заменяет сравнения итогового ролика на одинаковых noise/conditioning/steps.
