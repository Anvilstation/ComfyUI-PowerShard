"""Нода «PowerShard MiniMax H3: Veda Sparse Attention». Без torch на host: только проверка файла и настроек.

MODEL (из «PowerShard MiniMax H3: Loader») -> MODEL с настройками Veda в
transformer_options; они уходят в каждый RPC forward, workers считают разреженную attention.
"""
from pathlib import Path

CATEGORY = "PowerShard/MiniMax H3"
FOLDER = "veda"
KNOWN = {
    "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors":
        "https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
    "minimax_h3_r2va_veda_preview_fp8.safetensors":
        "https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview",
}


def register_folder():
    """models/veda (тот же каталог, что у официальной ноды Veda-on-ComfyUI)."""
    import folder_paths
    path = str(Path(folder_paths.models_dir) / FOLDER)
    entry = folder_paths.folder_names_and_paths.get(FOLDER)
    if entry is None:
        folder_paths.folder_names_and_paths[FOLDER] = ([path], {".safetensors"})
    else:
        if path not in entry[0]:
            entry[0].append(path)
        if entry[1] is not None and hasattr(entry[1], "add"):
            entry[1].add(".safetensors")
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


def predictor_choices():
    import folder_paths
    register_folder()
    local = list(folder_paths.get_filename_list(FOLDER))
    return local + [name for name in KNOWN if name not in local]


def predictor_path(name):
    import folder_paths
    register_folder()
    path = folder_paths.get_full_path(FOLDER, name)
    if path is None:
        folder = folder_paths.get_folder_paths(FOLDER)[0]
        url = KNOWN.get(name)
        raise ValueError(f"Предиктор Veda {name!r} не найден в {folder}"
                         + (f". Скачайте его: {url}/resolve/main/{name} и положите туда" if url else ""))
    return path


def build_spec(path, shape, world, sequence_mode, generated_sparsity, reference_sparsity, full_attention_layers,
               full_attention_steps, backend):
    """Проверить предиктор/настройки и собрать spec для workers (JSON-совместимый dict)."""
    from .veda import bundle as veda_bundle
    from .veda import settings
    metadata = veda_bundle.read_header_metadata(path)
    layers, heads, head_dim = veda_bundle.geometry(metadata)
    model = (int(shape["num_layers"]), int(shape["num_attention_heads"]), int(shape["attention_head_dim"]))
    if (layers, heads, head_dim) != model:
        raise ValueError(f"Предиктор {Path(path).name} обучен для {layers} блоков x {heads} голов x {head_dim}, "
                         f"а модель H3 — {model[0]} x {model[1]} x {model[2]}")
    if world > 1 and sequence_mode != "ulysses":
        raise ValueError("Veda в PowerShard работает в sequence_mode=ulysses (у каждого GPU полная последовательность "
                         "своих голов). Выберите ulysses в «PowerShard: GPU / память / attention»")
    if backend not in settings.BACKENDS:
        raise ValueError(f"backend: {' / '.join(settings.BACKENDS)}")
    trained_generated, trained_reference = veda_bundle.trained_budgets(metadata)
    dense_layers = settings.parse_index_list(full_attention_layers, "full_attention_layers")
    missing = [i for i in dense_layers if i >= layers]
    if missing:
        raise ValueError(f"full_attention_layers: у модели блоки 0-{layers - 1}, {missing} нет")
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                generated=settings.parse_sparsity(generated_sparsity, "generated_sparsity", trained_generated),
                reference=settings.parse_sparsity(reference_sparsity, "reference_sparsity", trained_reference),
                dense_layers=dense_layers,
                dense_steps=settings.parse_index_list(full_attention_steps, "full_attention_steps"),
                backend=backend)


class PowerShardH3Veda:
    """Veda Sparse Attention (ICML 2026) для распределённого MiniMax H3: ~10% пар тайлов attention."""

    @classmethod
    def INPUT_TYPES(cls):
        from .veda.settings import BACKENDS, DEFAULT_BUDGET
        return {"required": {
            "model": ("MODEL", {"tooltip": "MODEL из «PowerShard MiniMax H3: Loader»"}),
            "predictor": (predictor_choices(), {"tooltip": "Файл предиктора Veda в models/veda"})},
            "optional": {
                "generated_sparsity": ("STRING", {"default": DEFAULT_BUDGET, "tooltip":
                    "Сколько тайлов ключей (по 128 токенов) оставляет каждый тайл запроса: число (\"32\" — по "
                    "умолчанию, не зависит от размера видео), процент пропуска (\"90%\") или \"trained\""}),
                "reference_sparsity": ("STRING", {"default": DEFAULT_BUDGET, "tooltip":
                    "То же для референсов (первый/последний кадр, guide, reference). \"0%\" — полная attention"}),
                "full_attention_layers": ("STRING", {"default": "", "tooltip":
                    "Блоки (с нуля) с полной attention, например \"0, 1, 47-49\""}),
                "full_attention_steps": ("STRING", {"default": "", "tooltip":
                    "Шаги sampler (с нуля) с полной attention, например \"0\""}),
                "backend": (list(BACKENDS), {"default": "auto", "tooltip":
                    "auto: self-test + замер на каждом GPU, одинаковый выбор на всех rank. triton-int8 — sm80+; "
                    "triton-fp16 — Triton на V100 и др.; torch-sdpa — без Triton"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, predictor, generated_sparsity="32", reference_sparsity="32", full_attention_layers="",
              full_attention_steps="", backend="auto"):
        if not hasattr(model, "with_veda") or not hasattr(model, "powershard_h3_shape"):
            raise ValueError("Veda: нужен MODEL из «PowerShard MiniMax H3: Loader» "
                             "(исходный H3 loader и native модели не поддерживаются)")
        from .devices import resolve_gpu_selection
        config = model.session.config
        world = len(resolve_gpu_selection(config.gpu_ids))
        spec = build_spec(predictor_path(predictor), model.powershard_h3_shape, world, config.sequence_mode,
                          generated_sparsity, reference_sparsity, full_attention_layers, full_attention_steps,
                          backend)
        return (model.with_veda(spec),)

    @classmethod
    def IS_CHANGED(cls, model, predictor, generated_sparsity="32", reference_sparsity="32", full_attention_layers="",
                   full_attention_steps="", backend="auto"):
        try:
            stat = Path(predictor_path(predictor)).stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
        except (ValueError, OSError):
            stamp = None
        return (predictor, stamp, generated_sparsity, reference_sparsity, full_attention_layers,
                full_attention_steps, backend)


VEDA_NODE_CLASS_MAPPINGS = {"PowerShardH3Veda": PowerShardH3Veda}
VEDA_NODE_DISPLAY_NAME_MAPPINGS = {"PowerShardH3Veda": "PowerShard MiniMax H3: Veda Sparse Attention"}
