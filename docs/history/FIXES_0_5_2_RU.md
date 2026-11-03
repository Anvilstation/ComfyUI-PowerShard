# Исправления 0.5.2 — 5 октября 2026

## 1. QwenProxy.model_lowvram

В присланном message.txt H3 sampler -> prepare_sampling -> load_models_gpu -> free_memory выгружает Qwen. QwenPatcher.unpatch_model вызывает native ModelPatcher.unpatch_model, где читается model.model_lowvram. Начальный proxy получает поля через ModelPatcher.__init__, но QwenPatcher.clone и DistributedQwenCLIP.with_mlp заменяли его новым QwenProxy ПОСЛЕ конструктора patcher. У нового объекта полей не было.

Каждый QwenProxy теперь имеет model_lowvram, lowvram_patch_counter, model_loaded_weight_memory, model_offload_buffer_memory и current_weight_patches_uuid. Проверены настоящие native LoadedModel.model_load/model_unload для исходного CLIP, clone и MLP variant с release/cpu_shards/keep. Это CPU проверка точного проблемного пути, не запуск вашего Qwen checkpoint на V100.

## 2. MLP выключен по умолчанию

H3PatchConfig, QwenConfig, H3 Loader, Qwen Loader и обе MLP nodes: default off. FP16 Safe остаётся включён в H3 Loader независимо от chunking. Новый workflow generator тоже экспортирует off, а acceptance CLI по умолчанию не включает auto.

Явно сохранённые auto/manual в вашем старом workflow остаются включёнными — поменяйте mode на off или уберите MLP node. Не скрываем ваши настройки при миграции. Полный MLP может увеличить peak VRAM и дать реальный OOM; тогда нужен явный auto/manual.

## 3. H3 Loader: keep_in_memory

В Loader добавлен BOOLEAN keep_in_memory, default true. Он реально меняет Session.config.release_after_sampling. IS_CHANGED включает переключатель. Дублирующий keep_workers убран из Advanced UI, сохранён только legacy Python parsing. Schema 6 переносит известные старые значения в Loader без сдвига strict_attention/allow_host_wrappers/pin_memory.

| Событие | H3 CPU, keep=true | keep=false |
|---|---|---|
| Успешный sampling | end_run + idle; CPU shards сохранены | workers закрыты |
| ComfyUI unload / другая GPU-фаза | idle, CPU shards сохранены | workers закрыты |
| PowerShardRelease перед VAE | сохраняет CPU shards, если preserve_h3_cpu_shards=true | workers закрыты |
| Отмена здорового RPC | текущий RPC завершается в фоне, затем end_run/idle | workers закрыты |
| CUDA/NCCL/transport error | workers закрыты, повторная загрузка необходима | workers закрыты |

Для GPU/ATS keep=true сохраняет workers между обычными sampling calls, но явный Release или memory pressure освобождает их; на лету CUDA/managed weights не превращаются в CPUOffloadPolicy. CPU idle всё ещё оставляет небольшие CUDA buffers и NCCL context.

## 4. Почему ваш FA патчер не действовал

Присланный flash_attn.py не отдельное CUDA ядро: это Flash dense API поверх vllm_flash_attn.flash_attn_varlen_func. sys.path.insert в __init__ custom node изменяет только host process. Compatibility probes и worker subprocess имеют отдельный import path; при использовании установленного flash_attn они могут получить ошибку flash_attn_2_cuda, даже если host находит локальный shim. Фактическое место вашего файла в /opt не дано, поэтому конкретный origin надо читать в provider identity, а не угадывать.

В исходной dense обёртке Q/K/V reshape используют длину Q для K/V; H3 token sequence требует разных local Q и global K/V. Кроме того softmax_scale, dropout/window/ALiBi/deterministic/return flags принимались, но не передавались. Дефолтная шкала 1/sqrt(D) не заменяет требуемую после FP16 Safe Q/K normalization шкалу.

Добавлен powershard/flash_attn_shim.py, scoped исключительно для PowerShard. При выборе flash_attn сначала импортируется установленный provider; при ImportError локальный shim предоставляет API поверх существующего vLLM varlen kernel. Если native FA импортируется, но его numerical probe не проходит, обычная policy при разрешённом fallback проверяет vllm_flash_attn/SDPA. Никаких pip install, глобальных sys.path вставок или записи sys.modules['flash_attn'] нет.

У shims с реальным varlen export предпочтён этот entrypoint: он сохраняет отдельные Q/K offsets, GQA и explicit softmax_scale, обходя broken dense reshape. В identity записаны origin, entrypoint_module и implementation; выбранный flash_attn может означать vLLM-backed shim, а не независимый FA kernel. Numerical probe всё равно обязателен для всех выбранных GPU и padded-head Ulysses geometry.

Для явного использования собственного локального .py можно задать перед запуском ComfyUI:

```bash
export POWERSHARD_FLASH_ATTN_SHIM=/absolute/path/to/flash_attn.py
```

Это опционально: исправленный bundled shim уже включён. Относительный или отсутствующий путь отвергается. Стат файла входит в provider stamp; изменение provider требует перезапуска worker Session. Для прозрачного выбора непосредственно вашего CUDA implementation используйте attention_backend=vllm_flash_attn. Блок import os/sys; sys.path.insert из сообщения для этого пакета не нужен.

## 5. Почему отмена раньше выгружала ~50 GB

Session.call раньше делал close при ЛЮБОМ исключении, а H3 native unload, cleanup и PowerShardRelease тоже закрывали workers. Постоянные CPU weights принадлежат subprocess, не host proxy: остановка процесса уничтожала RAM shards. Это не было просто освобождением VRAM.

Теперь Session reuse переиспользует идентичного владельца при повторном исполнении Loader/patch nodes. Изменение checkpoint/stat/config/patch/provider запускает новую загрузку. При смене владельца той же роли старые workers закрываются, чтобы не хранить две H3-копии в 128 GB RAM. H3 и Qwen — отдельные роли; их CPU memory складывается.

Отмена уже запущенного здорового RPC с keep=true возвращает ComfyUI его исходное interrupt exception сразу. В фоне выполняется только оставшаяся часть этого RPC, результат отбрасывается, затем end_run и CPU idle. Входные и stage файлы сохраняются до всех rank acknowledgements; уже полученные ответы рангов не теряются. Новая GPU-фаза ждёт завершения drain. Status UI показывает draining и retained CPU idle.

Это не принудительное мгновенное прекращение GPU kernels. Безопасно сохранять FSDP/NCCL state после SIGTERM отдельных рангов нельзя. Ошибка kernel/collective/protocol закрывает Session; загрузка, прерванная до первого RPC, не сохраняется. Если kernel/collective завис, drain может не закончиться: принудительный restart ComfyUI освободит и RAM. Нет нового скрытого таймаута генерации.

## Проверка и ограничения

Результаты всего набора: **260 passed, 16 skipped** (45.53 s), reports/lifecycle-0.5.2/pytest.xml и verification.json. 16 real distributed CPU tests пропущены из-за запрета Gloo TCP transport в среде; аппаратный CUDA путь отдельно NOT_RUN. Ruff F821/F823, compileall, JavaScript syntax и git diff whitespace проходят. Проверены native Qwen load/unload, MLP defaults, переключатель Loader, migration schema 5 -> 6, reuse, отказ сохранять CUDA failure, отмена с частично полученными ответами 1/3 рангов и реальный CPU subprocess с tiny native H3: после Comfy interrupt PID сохранился, следующий RPC успешно выполнен. Mock rank queues подтверждают протокол, не NCCL.

Проверка FA — CPU reference с fake varlen provider: rectangular BLHD, batch>1, GQA, explicit scales, dense compatibility, origin и отсутствие глобального shadowing. Реальные FA/V100 CUDA kernels, POWER9, NCCL/FSDP на GPU, H3/Qwen production checkpoints, ATS, время итерации и качество генерации здесь НЕ ЗАПУСКАЛИСЬ. Старые performance-отчёты относятся к 0.5.1.

В message.txt также много `lora key not loaded`: соответствующие LoRA tensors не применены. PowerShard пока не реализует remote LoRA/ControlNet — исправление model_lowvram не добавляет эту поддержку. Для проверочного запуска уберите неподдерживаемую LoRA из workflow.

## Что проверить на сервере

Полностью перезапустите ComfyUI и обновите браузер после замены единственной папки custom node. Сначала прежний workflow: 5 GPU, cpu, token, H3 keep_in_memory=true, FP16 Safe=true, debug_finite=false, MLP off. Не меняйте prompt/shape/seed в одном сравнении. Выбор FA/vLLM проверьте в Status по actual calls и implementation/origin, а не только Requested.

Запустите первую генерацию, отмените после начала RPC, дождитесь idle и запустите тот же workflow. В output/powershard сравните PIDs/preflight/load_s и Session status: прежние workers должны сохраниться без повторной загрузки весов. Повторите со снятым keep — workers должны завершиться. Перед VAE оставьте preserve_h3_cpu_shards=true, если хотите сохранение RAM; для полного освобождения выключите оба preserve-флага.

Standalone CUDA probe, без переустановки окружения:

```bash
python scripts/probe_accelerators.py --gpus 0,1,2,3,5 --providers flash_attn,vllm_flash_attn,sdpa --world 5 --mode token --output reports/local-fa-052.json
```

Сначала проверьте --help: checkpoint geometry задаётся head_dim/heads/tokens, defaults соответствуют прежнему H3 case, а не универсальной модели. Probe выполняйте при свободных workers; он не измеряет FSDP/NCCL/H2D. Скорость и энергопотребление оценивайте отдельно от этих API/correctness исправлений.
