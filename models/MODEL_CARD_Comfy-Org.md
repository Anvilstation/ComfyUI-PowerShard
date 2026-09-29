---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
tags:
- comfyui
- diffusion-single-file
base_model:
- MiniMaxAI/MiniMax-H3
---

# MiniMax H3

Repackaged model files for ComfyUI.

Original model repository:

- https://huggingface.co/MiniMaxAI/MiniMax-H3
- https://huggingface.co/lightx2v/Minimax-h3-Turbo
- https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union
- https://huggingface.co/Kijai/MiniMax-H3-experimental

The Qwen3-VL-32B `nvfp4_awq` quant is converted from: https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4

This `nvfp4` text encoder does not require Blackwell GPU to use.

For diffusion models prefer `int8_convrot` if you are able to use pytorch with cu130.

`fp8_scaled` should only be used if you cannot use `int8_convrot`.

---

Place the files in the following folders:

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   ├── minimax_h3_fl2va_bf16.safetensors
│   │   ├── minimax_h3_fl2va_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_bf16.safetensors
│   │   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors
│   │   ├── minimax_h3_ref2va_bf16.safetensors
│   │   ├── minimax_h3_ref2va_int8_convrot.safetensors
│   │   ├── minimax_h3_ref2va_pruned_bf16.safetensors
│   │   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   │   └── minimax_h3_ref2va_pruned_fp8_scaled.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3vl_32b_minimax_h3_bf16.safetensors
│   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 loras/
│   │   ├── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
│   │   ├── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
│   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_audio_vae_fp32.safetensors
│   │   └── minimax_h3_video_vae_fp16.safetensors
│   ├── 📂 model_patches/
│   │   ├── minimax_h3_fun_controlnet_union_pruned_bf16.safetensors
│   │   └── minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors
│   ├── 📂 embeddings/
│   │   ├── minimaxh3_art_is_explosion.safetensors
│   │   ├── minimaxh3_blooming_flowers.safetensors
│   │   ├── minimaxh3_bullet_time.safetensors
│   │   ├── minimaxh3_dark_magic.safetensors
│   │   ├── minimaxh3_fire_breath.safetensors
│   │   ├── minimaxh3_four_seasons.safetensors
│   │   ├── minimaxh3_kiss_camera.safetensors
│   │   ├── minimaxh3_spiral_ascent.safetensors
│   │   ├── minimaxh3_storm_magic.safetensors
│   │   └── minimaxh3_truman_show.safetensors
```

## Workflows

- I2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json
- T2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json
- R2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json

Doc: https://docs.comfy.org/tutorials/video/minimax/minimax-h3

## Embeddings

To use the embeddings in a text prompt, invoke them in the CLIPTextEncode node by
combining `embedding:` with the filename of the embedding like this for `minimaxh3_art_is_explosion`:
```
embedding:minimaxh3_art_is_explosion
```
Just exchange the `minimaxh3_art_is_explosion` part with the filename of the embedding you want to use.