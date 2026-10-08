#!/usr/bin/env python3
"""Детерминированные API графы LTX-2 / 2.3 / 2.5 (workflows_ltx/). Имена файлов моделей — замените на свои."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows_ltx"
CKPT = "checkpoints/ltx-2.3-22b-dev.safetensors"
GEMMA = "gemma_3_12B_it.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2.safetensors"
NEGATIVE = ("blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles, "
            "distorted, deformed, disfigured, bad anatomy, static, jitter")
PROMPT = ("A lighthouse keeper climbs the spiral stairs at dusk, his lantern swinging; waves crash on the rocks outside "
          "and the wind howls through the window. He says quietly: \"Another storm tonight.\"")


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def common(width=768, height=512, length=121, fps=24.0, placement="gpu", lora=None, cache=False):
    g = {
        "1": node("PowerShardConfig", gpu_ids="all", weight_placement=placement, precision="fp16", attention_backend="auto",
                  sequence_mode="ulysses"),
        "2": node("PowerShardConfigTuning", config=["1", 0], reserve_gib=1.5, prefetch_blocks=0, numa_policy="auto",
                  strict_attention=False, allow_host_wrappers=False, pin_memory=True, sequence_comm_dtype="fp32",
                  prefetch_policy="auto"),
        "3": node("PowerShardLTXOptions", fp16_safe=False, debug_finite=False, mlp_chunk_mode="auto", mlp_chunk_tokens=4096,
                  batch_chunk=0, attention_chunk=8192),
        "4": node("PowerShardLTXLoader", checkpoint=CKPT, config=["2", 0], keep_in_memory=True, options=["3", 0]),
        "5": node("PowerShardLTXTextEncoder", text_encoder=GEMMA, ckpt_name=CKPT, config=["2", 0], after_encode="release"),
        "6": node("CLIPTextEncode", clip=["5", 0], text=PROMPT),
        "7": node("CLIPTextEncode", clip=["5", 0], text=NEGATIVE),
        "8": node("LTXVConditioning", positive=["6", 0], negative=["7", 0], frame_rate=fps),
        "9": node("PowerShardLTXVAELoader", ckpt_name=CKPT),
        "10": node("EmptyLTXVLatentVideo", width=width, height=height, length=length, batch_size=1),
        "11": node("LTXVEmptyLatentAudio", frames_number=length, frame_rate=fps, batch_size=1, audio_vae=["9", 1]),
        "30": node("KSamplerSelect", sampler_name="euler"),
        "31": node("RandomNoise", noise_seed=42),
    }
    if lora:
        g["40"] = node("PowerShardLTXLoRA", lora_name=lora, strength=1.0)
        g["4"]["inputs"]["lora"] = ["40", 0]
    model = ["4", 0]
    if cache:
        g["41"] = node("PowerShardBlockCache", model=model, threshold=0.08, start_percent=0.15, end_percent=0.95,
                       max_consecutive_skips=3, warmup_steps=2)
        model = ["41", 0]
    g["42"] = node("LTXVSpatioTemporalGuidance", model=model, scale=1.0, blocks="29", start_percent=0.0, end_percent=1.0)
    g["43"] = node("LTXVModalityGuidance", model=["42", 0], modality_scale=3.0, start_percent=0.0, end_percent=1.0)
    return g, ["43", 0]


def decode(g, latent, fps, prefix):
    g["50"] = node("LTXVSeparateAVLatent", av_latent=latent)
    g["51"] = node("VAEDecodeTiled", samples=["50", 0], vae=["9", 0], tile_size=512, overlap=64, temporal_size=64,
                   temporal_overlap=8)
    g["52"] = node("LTXVAudioVAEDecode", samples=["50", 1], audio_vae=["9", 1])
    g["53"] = node("CreateVideo", images=["51", 0], fps=fps, audio=["52", 0])
    g["54"] = node("SaveVideo", video=["53", 0], filename_prefix=f"video/{prefix}", format="auto", codec="auto")
    return g


def sample(g, model, positive, negative, latent, steps=30, video_cfg=3.0, audio_cfg=7.0, sigmas=None, key="20"):
    g[key] = node("LTXVConcatAVLatent", video_latent=latent[0], audio_latent=latent[1]) if isinstance(latent, tuple) else None
    if g[key] is None:
        del g[key]
        av = latent
    else:
        av = [key, 0]
    if sigmas is None:
        g[key + "1"] = node("LTXVScheduler", steps=steps, max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1, latent=av)
    else:
        g[key + "1"] = node("ManualSigmas", sigmas=sigmas)
    g[key + "2"] = node("LTXVDualCFGGuider", model=model, positive=positive, negative=negative, video_cfg=video_cfg,
                        audio_cfg=audio_cfg)
    g[key + "3"] = node("SamplerCustomAdvanced", noise=["31", 0], guider=[key + "2", 0], sampler=["30", 0],
                        sigmas=[key + "1", 0], latent_image=av)
    return [key + "3", 0]


def t2v():
    g, model = common()
    out = sample(g, model, ["8", 0], ["8", 1], (["10", 0], ["11", 0]))
    return decode(g, out, 24.0, "ltx_t2v_audio")


def i2v():
    g, model = common()
    g["60"] = node("LoadImage", image="example.png")
    g["61"] = node("LTXVPreprocess", image=["60", 0])
    g["62"] = node("LTXVImgToVideoInplace", vae=["9", 0], image=["61", 0], latent=["10", 0], strength=1.0, bypass=False)
    out = sample(g, model, ["8", 0], ["8", 1], (["62", 0], ["11", 0]))
    return decode(g, out, 24.0, "ltx_i2v_audio")


def keyframes():
    """Первый и последний кадр (guides) + IC-LoRA: guide-токены и маска внимания считаются в workers."""
    g, model = common(lora="ltx-2.3-22b-ic-lora-union-control.safetensors")
    g["60"] = node("LoadImage", image="first.png")
    g["61"] = node("LoadImage", image="last.png")
    g["62"] = node("LTXVAddGuide", positive=["8", 0], negative=["8", 1], vae=["9", 0], latent=["10", 0], image=["60", 0],
                   frame_idx=0, strength=1.0)
    g["63"] = node("LTXVAddGuide", positive=["62", 0], negative=["62", 1], vae=["9", 0], latent=["62", 2], image=["61", 0],
                   frame_idx=-1, strength=0.8)
    out = sample(g, model, ["63", 0], ["63", 1], (["63", 2], ["11", 0]))
    g["64"] = node("LTXVSeparateAVLatent", av_latent=out)
    g["65"] = node("LTXVCropGuides", positive=["63", 0], negative=["63", 1], latent=["64", 0])
    g["66"] = node("LTXVConcatAVLatent", video_latent=["65", 2], audio_latent=["64", 1])
    return decode(g, ["66", 0], 24.0, "ltx_keyframes_iclora")


def two_stage():
    """Низкое разрешение -> латентный x2 апскейлер -> короткий второй проход (distilled-сигмы)."""
    g, model = common(width=640, height=384, length=121)
    first = sample(g, model, ["8", 0], ["8", 1], (["10", 0], ["11", 0]), steps=20)
    g["70"] = node("LTXVSeparateAVLatent", av_latent=first)
    g["71"] = node("LatentUpscaleModelLoader", model_name=UPSCALER)
    g["72"] = node("LTXVLatentUpsampler", samples=["70", 0], upscale_model=["71", 0], vae=["9", 0])
    second = sample(g, model, ["8", 0], ["8", 1], (["72", 0], ["70", 1]), video_cfg=1.0, audio_cfg=1.0,
                    sigmas="0.909375, 0.725, 0.421875, 0.0", key="80")
    return decode(g, second, 24.0, "ltx_two_stage_x2")


def long_video():
    """Длинное видео: context windows (родной узел) + PowerShard Block Cache."""
    g, model = common(length=481, cache=True)
    g["90"] = node("ContextWindowsManual", model=model, context_length=32, context_overlap=8,
                   context_schedule="standard_static", context_stride=1, closed_loop=False, fuse_method="pyramid", dim=2,
                   freenoise=False, cond_retain_index_list="", split_conds_to_windows=False, latent_retain_index_list="",
                   causal_window_fix=True)
    out = sample(g, ["90", 0], ["8", 0], ["8", 1], (["10", 0], ["11", 0]))
    return decode(g, out, 24.0, "ltx_long_context_windows")


def main():
    OUT.mkdir(exist_ok=True)
    for name, build in (("ltx2_t2v_audio", t2v), ("ltx2_i2v_audio", i2v), ("ltx2_keyframes_iclora", keyframes),
                        ("ltx2_two_stage_upscale", two_stage), ("ltx2_long_video_context_windows", long_video)):
        (OUT / f"{name}.api.json").write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
