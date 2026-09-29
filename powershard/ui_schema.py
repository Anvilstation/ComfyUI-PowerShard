"""Русские подписи/подсказки, общий источник для native tooltip и панели UI.

Не импортирует torch, ComfyUI или optional kernels. Значения enum и имена
сокетов не переводятся в сериализованном workflow.
"""
from copy import deepcopy

# label, help, group
FIELDS = {
    "gpu_ids": ("Выбранные GPU", "Индексы GPU, видимые исходному процессу ComfyUI. Порядок сохраняется: 5,2,0 → ranks 0,1,2. all выбирает все видимые GPU. Одна карта допустима, но это не межкарточное шардирование.", "Основное"),
    "backend": ("Распределение работы", "FSDP2 распределяет веса. FSDP2 + sequence дополнительно делит вычисления по токенам; выигрыш зависит от длины и обменов. На коротких задачах может быть медленнее.", "Основное"),
    "precision": ("Хранение весов", "FP16: floating checkpoint с FP16 вычислениями и FP32 безопасными участками. INT8: ConvRot хранится квантованным, активные строки деквантизуются для FP16 GEMM. Нужен соответствующий файл весов.", "Основное"),
    "checkpoint": ("Файл модели", "Локальный checkpoint в diffusion_models либо text_encoders. Нода не скачивает и не изменяет исходный файл.", "Основное"),
    "config": ("Конфигурация PowerShard", "GPU, размещение памяти и attention для этой модели. H3 и Qwen могут иметь отдельные Config-ноды.", "Основное"),
    "model": ("Модель PowerShard", "MODEL из загрузчика PowerShard. Patcher создаёт независимую конфигурацию ветки и передаёт её реальным worker-процессам.", "Основное"),
    "memory_profile": ("Профиль памяти", "Минимум VRAM: CPU-шарды, prefetch=0, меньшие FSDP-группы attention/MLP, MLP auto, без prepared-весов, conditioning/Spectrum-кэш в RAM, SDPA частями. Переопределяет cpu_offload/prefetch/memory_policy и MLP mode. Не меняет выбранный attention. Может существенно замедлить работу. Пользовательский сохраняет прежнее поведение.", "Память"),
    "workspace_mib": ("Временный буфер, MiB", "В профиле Минимум VRAM ограничивает оценку временного MLP/attention буфера. Это не общий лимит VRAM: активные веса, полный output/KV, CUDA/NCCL и внутренние kernel buffers добавляются отдельно. Если даже минимальная порция не помещается, OOM остаётся возможен.", "Память"),
    "stage_cache_mib": ("Кэш conditioning в RAM, MiB", "Лимит CPU-кэша conditioning каждого H3 worker в профиле Минимум VRAM. 0 отключает кэш, чтение повторяется. Текущие входы всё равно переносятся на GPU для вычисления.", "Память"),
    "reserve_gib": ("Резерв VRAM, GiB", "Планировщик оставляет этот объём в оценке для внешних аллокаций. Не резервирует физическую память и не запрещает запуск по прогнозу OOM.", "Память"),
    "cpu_offload": ("Шарды весов в RAM", "CPUOffloadPolicy хранит локальные FSDP-шарды на CPU, подаёт активную группу на GPU и освобождает её после forward. Это не выгрузка всех activations. Минимум VRAM включает автоматически.", "Память"),
    "weight_placement": ("Размещение / ATS диагностика", "gpu учитывает галочку CPU offload; cpu включает CPUOffloadPolicy. ats включает тот же offload и отдельную диагностику capabilities/передач. ATS allocator модели НЕ реализован; это не защита от любого OOM.", "Дополнительно"),
    "pin_memory": ("Закреплять CPU-память", "Позволяет асинхронную передачу CPU-шардов. Закреплённая RAM недоступна swap и может быть дороже для системы. Сравните on/off на вашем NUMA/NVLink стенде.", "Память"),
    "prefetch_blocks": ("Загрузить блоки заранее", "0: только активные группы; 1/2: подготовка следующих блоков с большим расходом VRAM. В профиле Минимум VRAM всегда 0. Ускорение требует измерений.", "Память"),
    "memory_policy": ("Планирование MLP / prefetch", "manual использует заданные пределы. auto оценивает свободную память перед forward и согласует бюджет между ranks. Оценка не гарантирует отсутствие OOM.", "Дополнительно"),
    "numa_policy": ("NUMA размещение worker", "none не меняет affinity. auto привязывает CPU-потоки к близкой GPU NUMA-области, если она определена. bind дополнительно запрашивает привязку CPU-памяти; результат проверяется. Не доказывает использование NVLink.", "Дополнительно"),
    "attention_backend": ("Вычисление attention", "auto выбирает прошедший проверку доступный путь, не обещает самый быстрый. SDPA использует automatic dispatch. Flash/vLLM используют установленный пакет. Sage — явно выбранный квантованный режим; math — точное attention частями для сравнения, обычно медленнее. Минимум VRAM делит SDPA по Q/heads, не обрезая K/V.", "Основное"),
    "allow_fallback": ("Разрешить совместимый fallback", "Если optional provider не поддерживает вызов, используется путь с той же семантикой и записывается причина. OOM/illegal access не маскируются повтором. Выключите для диагностики конкретного kernel.", "Дополнительно"),
    "sequence_mode": ("Алгоритм sequence", "token делит токены и собирает полные K/V. ulysses делит heads и требует делимости числа heads на число GPU; иначе используется token на всех выбранных GPU. Для обычного FSDP2 эта настройка не действует.", "Дополнительно"),
    "sequence_comm_dtype": ("Передача sequence-тензоров", "FP32 сохраняет диапазон. H3 token FP16 передаёт K/V с согласованным power-of-two scaling. Ulysses QKV остаются native dtype. Qwen передаёт KV/hidden в FP32 независимо от этой настройки.", "Дополнительно"),
    "timeout_s": ("Таймаут worker, с", "Предел ожидания команды/collectives. Для загрузки больших моделей и медленного RAM-профиля может требоваться больше времени. Таймаут завершает сессию, а не оставляет зависшие ranks.", "Дополнительно"),
    "allow_unverified": ("Старое поле совместимости", "Оставлено для старых workflows. Больше не ограничивает запуск по версии или типу GPU; проверяются реальные возможности.", "Совместимость"),
    "release_after_sampling": ("Освободить workers после sampling", "Завершает H3 workers и освобождает их GPU-память перед следующей фазой. Повторная генерация потребует загрузки весов. При offload можно выключить для повторного использования CPU-шардов.", "Память"),
    "enabled": ("Включить patch", "Включает patch только в этой ветке MODEL. Изменение конфигурации создаёт соответствующую worker-сессию, не меняя глобальные классы ComfyUI.", "Основное"),
    "fp16_safe": ("Безопасный FP16", "FP32 condition/residual/norm и чувствительные операции, scaled FP16 GEMM для тяжёлых матриц. Не clamp и не nan_to_num. Рекомендуется для H3 на V100.", "Основное"),
    "debug_finite": ("Проверять NaN / Inf", "Отложенные finite-флаги проверяются на границе forward. Для диагностики; добавляет расходы. Не изменяет результат и не исправляет нечисловые значения.", "Дополнительно"),
    "mlp_chunk_mode": ("Обработка MLP частями", "manual: заданное число токенов; auto: по оценке памяти; off: вся локальная последовательность сразу, FP16 Safe остаётся. Минимум VRAM принудительно использует auto. Большие порции могут ускорять, но увеличивают пик VRAM.", "Память"),
    "mlp_chunk_tokens": ("Токенов в порции MLP", "Размер порции в manual. В auto выводится фактический размер; в off не действует. Например 16384 может быть быстрее 512 при достаточной памяти. Результат математически тот же, округления FP16 могут отличаться.", "Память"),
    "idle_policy": ("Qwen после кодирования", "release завершает encoder workers; cpu_shards сохраняет шардированные веса в RAM и освобождает CUDA-кэш; keep сохраняет сессию и её размещение. CPU shards требует offload. Минимум VRAM включает offload.", "Память"),
    "cache_mib": ("Готовые embeddings в RAM, MiB", "Кэш результата кодирования Qwen в основном процессе. 0 отключает. Повтор одинаковых prompt/images может обойти encoder, но смена модели/настроек меняет ключ.", "Память"),
    "placement": ("Родной encoder: размещение", "cpu_fp32 исполняет BF16 source на CPU в FP32: медленно, без распределения вычислений. native_offload использует управление памятью ComfyUI. Для распределённого Qwen выберите новый загрузчик Qwen.", "Основное"),
    "samples": ("Latent после sampler", "Поставьте Release между sampler и разделением video/audio latent, чтобы освободить память PowerShard перед VAE.", "Основное"),
    "preserve_qwen_cpu_shards": ("Оставить idle Qwen в RAM", "Не завершать Qwen-сессии, сохранённые через cpu_shards. Это ускоряет повторное кодирование, но удерживает RAM и небольшой CUDA/NCCL context.", "Память"),
    "clear_conditioning_cache": ("Очистить embeddings-кэш", "Удаляет сохранённые результаты кодирования в RAM. Следующее кодирование одинакового текста будет выполнено заново.", "Память"),
    "refresh": ("Повторить диагностику", "Измените число и запустите ноду для нового отчёта. Диагностика не устанавливает пакеты и не изменяет драйвер.", "Основное"),
    "history_device": ("История Spectrum", "cpu экономит VRAM ценой передач. cuda удерживает историю на GPU. Минимум VRAM переопределяет это в cpu; остальные параметры аппроксимации сохраняются.", "Память"),
    "history_mib": ("Лимит истории, MiB", "Ограничивает тензоры истории Spectrum. При нехватке выполняется настоящий forward. Это не общий лимит памяти модели.", "Память"),
    "degree": ("Степень прогноза", "Сложность полиномиального компонента Spectrum. Больше не означает лучше; сравнивайте исходные PNG и audio с выключенным Spectrum.", "Дополнительно"),
    "warmup": ("Настоящих начальных шагов", "Первые шаги для накопления истории. На few-step модели прогрев и завершающие шаги могут занять весь schedule — пропусков не будет.", "Основное"),
    "tail": ("Настоящих завершающих шагов", "Последние шаги выполняются без прогноза. Помогает ограничить накопленную ошибку, но не гарантирует качество.", "Основное"),
    "max_forecast": ("Прогнозов подряд", "Максимум пропусков DiT подряд. Повышение ускоряет только при приемлемой ошибке; это приближённый режим.", "Основное"),
    "history_size": ("Записей в истории", "Число последних состояний для прогноза. Ограничивается также history_mib.", "Дополнительно"),
    "ridge": ("Регуляризация прогноза", "Положительный коэффициент стабилизации решения Spectrum. Меняет аппроксимацию, проверяйте качество отдельно.", "Дополнительно"),
    "blend": ("Спектральная доля video", "Смешивание спектрального и линейного прогноза: 0 — линейный, 1 — спектральный. Даже 0 остаётся прогнозом, а не настоящим forward.", "Основное"),
    "audio_blend": ("Спектральная доля audio", "0 оставляет линейный прогноз audio. Увеличение может менять речь/музыку; сравнивайте аудио, а не только кадры.", "Основное"),
    "allow_any_sampler": ("Разрешить другие samplers", "Экспериментальный opt-in вне deterministic Euler. При stochastic/multistage solver качество не гарантировано. Ненулевой churn отключает прогноз.", "Дополнительно"),
}
NODES = {
 "PowerShardConfig": ("PowerShard · GPU и память", "Единая конфигурация для H3 и Qwen. Начните с выбора GPU, attention и профиля памяти. Панель «Параметры и пояснения» не меняет формат старых workflows."),
 "PowerShardH3Loader": ("PowerShard · загрузить H3", "Шардированный MiniMax H3 FL2VA/Ref2VA. Соедините MODEL с FP16 Patcher и обычным native H3 guider/sampler."),
 "PowerShardH3QwenLoader": ("PowerShard · загрузить Qwen H3", "Родной Qwen3VL-32B H3 с FSDP, опциональным CPU offload и sequence. Выход — совместимый CLIP; vision вычисляется реплицированно."),
 "PowerShardH3TextEncoder": ("PowerShard · родной encoder (legacy)", "Сохранён для старых графов. CPU FP32 либо стандартный offload ComfyUI; для FSDP используйте «загрузить Qwen H3»."),
 "PowerShardH3FP16Patcher": ("PowerShard · безопасный FP16 H3", "Patch применяется внутри всех workers до FSDP. Clone не изменяет исходный MODEL. В профиле Минимум VRAM MLP автоматически делится по бюджету."),
 "PowerShardRelease": ("PowerShard · освободить перед VAE", "Остановить H3 workers перед video/audio decoding; опционально сохранить Qwen CPU-шарды."),
 "PowerShardDiagnostics": ("PowerShard · диагностика", "Среда, устройства, топология и фактические возможности. Версия и архитектура сами по себе не запрещают работу."),
 "PowerShardSpectrum": ("PowerShard · Spectrum (приближение)", "Прогноз вместо некоторых DiT forward. Может ускорить и изменить результат. Для проверки качества сначала выключите; ATS и OOM protection это не заменяет."),
}
ENUMS = {
 "memory_profile": {"custom":"Пользовательский / прежнее поведение", "ram_min":"Минимум VRAM · веса в RAM"},
 "attention_backend": {"auto":"Авто · прошедший проверку", "sdpa":"PyTorch SDPA", "flash_attn":"FlashAttention · установленный пакет", "vllm_flash_attn":"FlashAttention · vLLM / custom build", "sageattention":"SageAttention · квантованный", "math":"Math · точное attention частями"},
 "backend": {"fsdp2":"FSDP2 · распределение весов", "fsdp2_sequence":"FSDP2 + sequence · веса и вычисления"},
 "precision": {"fp16":"FP16 + FP32 безопасные участки", "int8_fp16":"INT8 ConvRot · FP16 вычисления"},
 "weight_placement": {"gpu":"GPU / галочка offload", "cpu":"CPU-шарды · обычный offload", "ats":"CPU-шарды + ATS диагностика (не allocator)"},
 "mlp_chunk_mode": {"auto":"Авто · по бюджету", "manual":"Задать размер порции", "off":"Без разбиения MLP"},
 "idle_policy": {"release":"Завершить encoder workers", "cpu_shards":"Оставить шарды в RAM", "keep":"Сохранить сессию"},
 "numa_policy": {"none":"Не менять affinity", "auto":"CPU affinity по топологии", "bind":"CPU и memory binding"},
}


def schema():
    return dict(nodes={k:dict(title=v[0],description=v[1]) for k,v in NODES.items()},
                fields={k:dict(label=v[0],help=v[1],group=v[2]) for k,v in FIELDS.items()}, enums=deepcopy(ENUMS))


def annotate_nodes(classes, display):
    for name, cls in classes.items():
        title, description = NODES[name]
        display[name] = title
        cls.DESCRIPTION = description
        original = cls.INPUT_TYPES.__func__
        def inputs(owner, _original=original):
            result = _original(owner)
            for group in result.values():
                for key, spec in list(group.items()):
                    if key not in FIELDS:
                        continue
                    options = dict(spec[1]) if len(spec) > 1 else {}
                    options['tooltip'] = FIELDS[key][1]
                    group[key] = (spec[0], options, *spec[2:])
            return result
        cls.INPUT_TYPES = classmethod(inputs)
