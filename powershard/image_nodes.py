"""Ноды image-моделей (SDXL, Flux, Flux.2, Z-Image, Qwen-Image, MOSS-Image, ...). Lazy imports: без torch."""
CATEGORY = "PowerShard/Image"
PARALLEL = ["auto", "sequence", "batch", "replicate"]
PARALLEL_TIP = ("Как делить вычисление между GPU. sequence — токены картинки по GPU (Flux / Flux.2 / Chroma, "
                "Qwen-Image, Z-Image / Lumina 2, MOSS-Image): быстрее всего при batch 1. batch — строки batch "
                "(cond/uncond, batch_size) по GPU: любая модель, в т.ч. SDXL. replicate — все GPU считают одно и то же, "
                "делятся только веса. auto — sequence, если семейство поддерживает, иначе batch.")


class PowerShardImageLoader:
    """Image-модель любого семейства ComfyUI: веса шардируются по GPU (FSDP2), forward — родной код ComfyUI."""
    @classmethod
    def INPUT_TYPES(cls):
        from .ltx_nodes import _model_choices
        return {"required": {"checkpoint": (_model_choices(), {"tooltip": "diffusion_models/ (только модель) или "
                                                                        "checkpoints/ (полный checkpoint, напр. SDXL)"}),
                             "config": ("POWERSHARD_CONFIG",),
                             "parallel": (PARALLEL, {"default": "auto", "tooltip": PARALLEL_TIP})},
                "optional": {"keep_in_memory": ("BOOLEAN", {"default": True}),
                             "weights": ("POWERSHARD_WEIGHTS", {"tooltip": "Формат весов, dtype (auto: bf16 на sm80+, "
                                                                           "fp16 на V100; bf16-модели на V100 — "
                                                                           "bf16 хранение + FP32 вычисление)"}),
                             "lora": ("POWERSHARD_LORA",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, checkpoint, config, parallel="auto", keep_in_memory=True, weights=None, lora=None):
        from dataclasses import replace
        from .image_adapter import load_image
        from .ltx_nodes import _resolve
        from .wan_nodes import _report_dir
        config = replace(config, release_after_sampling=not keep_in_memory)
        return (load_image(str(_resolve(checkpoint)), config, parallel, weights, tuple(lora or ()), _report_dir()),)

    @classmethod
    def IS_CHANGED(cls, checkpoint, config, parallel="auto", keep_in_memory=True, weights=None, lora=None):
        from .ltx_nodes import _choice_stat
        from .web_api import provider_stamp
        return (_choice_stat(checkpoint), repr(config), parallel, keep_in_memory, repr(weights), repr(lora),
                provider_stamp())


class PowerShardCheckpointCLIPVAE:
    """CLIP и VAE из полного checkpoint (SDXL, SD 1.5, ...) без загрузки UNet на host."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"),)}}
    RETURN_TYPES = ("CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, ckpt_name):
        import comfy.sd
        import folder_paths
        path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(path, output_vae=True, output_clip=True, output_model=False,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        if out[1] is None or out[2] is None:
            raise ValueError(f"{ckpt_name}: в checkpoint нет CLIP или VAE — загрузите их отдельными нодами")
        return (out[1], out[2])

    @classmethod
    def IS_CHANGED(cls, ckpt_name):
        from .wan_nodes import _stat
        return _stat("checkpoints", ckpt_name)


IMAGE_NODE_CLASS_MAPPINGS = {
    "PowerShardImageLoader": PowerShardImageLoader,
    "PowerShardCheckpointCLIPVAE": PowerShardCheckpointCLIPVAE,
}
IMAGE_NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerShardImageLoader": "PowerShard Image: Loader (SDXL / Flux / Z-Image / Qwen / MOSS ...)",
    "PowerShardCheckpointCLIPVAE": "PowerShard Image: CLIP + VAE из checkpoint (без UNet)",
}
