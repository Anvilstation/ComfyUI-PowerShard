#!/usr/bin/env python3
"""Детерминированные API/UI графы Wan 2.2 (workflows_wan/). H3 workflows/ не трогаются."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows_wan"
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
            "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
# class: (sockets [(name,type)], outputs, widgets in UI order, extra UI widget values after a name)
SPECS = {
    "PowerShardConfig": ([], ["POWERSHARD_CONFIG"], ["gpu_ids", "weight_placement", "precision", "attention_backend", "sequence_mode"]),
    "PowerShardConfigTuning": ([("config", "POWERSHARD_CONFIG")], ["POWERSHARD_CONFIG"],
                               ["reserve_gib", "prefetch_blocks", "numa_policy", "strict_attention", "allow_host_wrappers",
                                "pin_memory", "sequence_comm_dtype", "prefetch_policy"]),
    "PowerShardWanOptions": ([], ["WAN_OPTIONS"], ["fp16_safe", "debug_finite", "mlp_chunk_mode", "mlp_chunk_tokens",
                                                    "moe_residency", "batch_chunk", "model_type"]),
    "PowerShardWanLoRA": ([("lora", "WAN_LORA")], ["WAN_LORA"], ["lora_name", "strength", "apply_to"]),
    "PowerShardWan22MoELoader": ([("config", "POWERSHARD_CONFIG"), ("options", "WAN_OPTIONS"), ("lora", "WAN_LORA")],
                                 ["MODEL", "MODEL", "MODEL"],
                                 ["high_noise_checkpoint", "low_noise_checkpoint", "boundary", "keep_in_memory"]),
    "PowerShardWanLoader": ([("config", "POWERSHARD_CONFIG"), ("options", "WAN_OPTIONS"), ("lora", "WAN_LORA")], ["MODEL"],
                            ["checkpoint", "keep_in_memory"]),
    "PowerShardWanTextEncoder": ([], ["CLIP"], ["checkpoint", "placement"]),
    "PowerShardWanT5Distributed": ([("config", "POWERSHARD_CONFIG")], ["CLIP"], ["checkpoint", "after_encode"]),
    "CLIPTextEncode": ([("clip", "CLIP")], ["CONDITIONING"], ["text"]),
    "VAELoader": ([], ["VAE"], ["vae_name"]),
    "LoadImage": ([], ["IMAGE", "MASK"], ["image"]),
    "ModelSamplingSD3": ([("model", "MODEL")], ["MODEL"], ["shift"]),
    "EmptyHunyuanLatentVideo": ([], ["LATENT"], ["width", "height", "length", "batch_size"]),
    "WanImageToVideo": ([("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
                         ("start_image", "IMAGE")], ["CONDITIONING", "CONDITIONING", "LATENT"],
                        ["width", "height", "length", "batch_size"]),
    "Wan22ImageToVideoLatent": ([("vae", "VAE"), ("start_image", "IMAGE")], ["LATENT"], ["width", "height", "length", "batch_size"]),
    "KSamplerAdvanced": ([("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                         ["LATENT"], ["add_noise", "noise_seed", "steps", "cfg", "sampler_name", "scheduler",
                                      "start_at_step", "end_at_step", "return_with_leftover_noise"]),
    "KSampler": ([("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                 ["LATENT"], ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]),
    "PowerShardRelease": ([("samples", "LATENT")], ["LATENT", "STRING"],
                          ["preserve_qwen_cpu_shards", "clear_conditioning_cache", "preserve_h3_cpu_shards"]),
    "VAEDecodeTiled": ([("samples", "LATENT"), ("vae", "VAE")], ["IMAGE"], ["tile_size", "overlap", "temporal_size", "temporal_overlap"]),
    "CreateVideo": ([("images", "IMAGE")], ["VIDEO"], ["fps"]),
    "SaveVideo": ([("video", "VIDEO")], [], ["filename_prefix", "format", "codec"]),
}
SEED_CONTROL = {"KSamplerAdvanced": "noise_seed", "KSampler": "seed"}


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def common(placement="gpu", mode="ulysses", comm="fp32", residency="swap", mlp="auto", batch_chunk=0, prefetch=0):
    return {
        "1": node("PowerShardConfig", gpu_ids="all", weight_placement=placement, precision="fp16",
                  attention_backend="auto", sequence_mode=mode),
        "2": node("PowerShardConfigTuning", config=["1", 0], reserve_gib=1.5, prefetch_blocks=prefetch, numa_policy="auto",
                  strict_attention=False, allow_host_wrappers=False, pin_memory=True, sequence_comm_dtype=comm,
                  prefetch_policy="auto"),
        "3": node("PowerShardWanOptions", fp16_safe=False, debug_finite=False, mlp_chunk_mode=mlp, mlp_chunk_tokens=4096,
                  moe_residency=residency, batch_chunk=batch_chunk, model_type="auto"),
        # umT5 шардируется по тем же GPU (FP32 вычисление) и закрывается после encode: cuda:0 не забивается.
        "5": node("PowerShardWanT5Distributed", checkpoint="umt5_xxl_fp16.safetensors", config=["2", 0], after_encode="release"),
        "6": node("CLIPTextEncode", clip=["5", 0], text="Камера медленно облетает маяк на скалистом берегу на закате, "
                  "волны разбиваются о камни, кинематографичный свет."),
        "7": node("CLIPTextEncode", clip=["5", 0], text=NEGATIVE),
    }


def tail(graph, samples, vae, fps, name):
    graph["20"] = node("PowerShardRelease", samples=samples, preserve_qwen_cpu_shards=True, clear_conditioning_cache=False,
                       preserve_h3_cpu_shards=True)
    graph["21"] = node("VAEDecodeTiled", samples=["20", 0], vae=vae, tile_size=512, overlap=64, temporal_size=64,
                       temporal_overlap=8)
    graph["22"] = node("CreateVideo", images=["21", 0], fps=float(fps))
    graph["23"] = node("SaveVideo", video=["22", 0], filename_prefix="video/" + name, format="auto", codec="auto")
    return graph


def moe_two_samplers(name, high, low, vae="wan_2.1_vae.safetensors", width=832, height=480, length=81, steps=20, split=10,
                     cfg=3.5, shift=8.0, boundary=0.875, image=False, loras=(), **kw):
    graph = common(**kw)
    lora_ref = None
    for index, (file, apply_to) in enumerate(loras):
        key = str(30 + index)
        inputs = dict(lora_name=file, strength=1.0, apply_to=apply_to)
        if lora_ref:
            inputs["lora"] = lora_ref
        graph[key] = node("PowerShardWanLoRA", **inputs)
        lora_ref = [key, 0]
    loader = dict(high_noise_checkpoint=high, low_noise_checkpoint=low, config=["2", 0], boundary=boundary,
                  keep_in_memory=True, options=["3", 0])
    if lora_ref:
        loader["lora"] = lora_ref
    graph["4"] = node("PowerShardWan22MoELoader", **loader)
    graph["8"] = node("VAELoader", vae_name=vae)
    graph["10"] = node("ModelSamplingSD3", model=["4", 0], shift=shift)
    graph["11"] = node("ModelSamplingSD3", model=["4", 1], shift=shift)
    if image:
        graph["9"] = node("LoadImage", image="example.png")
        graph["12"] = node("WanImageToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], start_image=["9", 0],
                           width=width, height=height, length=length, batch_size=1)
        positive, negative, latent = ["12", 0], ["12", 1], ["12", 2]
    else:
        graph["12"] = node("EmptyHunyuanLatentVideo", width=width, height=height, length=length, batch_size=1)
        positive, negative, latent = ["6", 0], ["7", 0], ["12", 0]
    graph["13"] = node("KSamplerAdvanced", model=["10", 0], positive=positive, negative=negative, latent_image=latent,
                       add_noise="enable", noise_seed=42, steps=steps, cfg=cfg, sampler_name="euler", scheduler="simple",
                       start_at_step=0, end_at_step=split, return_with_leftover_noise="enable")
    graph["14"] = node("KSamplerAdvanced", model=["11", 0], positive=positive, negative=negative, latent_image=["13", 0],
                       add_noise="disable", noise_seed=42, steps=steps, cfg=cfg, sampler_name="euler", scheduler="simple",
                       start_at_step=split, end_at_step=10000, return_with_leftover_noise="disable")
    return tail(graph, ["14", 0], ["8", 0], 16, name)


def moe_single_sampler(name, high, low, steps=20, cfg=3.5, shift=8.0, boundary=0.875, **kw):
    graph = common(**kw)
    graph["4"] = node("PowerShardWan22MoELoader", high_noise_checkpoint=high, low_noise_checkpoint=low, config=["2", 0],
                      boundary=boundary, keep_in_memory=True, options=["3", 0])
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["10"] = node("ModelSamplingSD3", model=["4", 2], shift=shift)
    graph["12"] = node("EmptyHunyuanLatentVideo", width=832, height=480, length=81, batch_size=1)
    graph["13"] = node("KSampler", model=["10", 0], positive=["6", 0], negative=["7", 0], latent_image=["12", 0],
                       seed=42, steps=steps, cfg=cfg, sampler_name="euler", scheduler="simple", denoise=1.0)
    return tail(graph, ["13", 0], ["8", 0], 16, name)


def ti2v_5b(name, image=True, width=1280, height=704, length=121, steps=30, cfg=5.0, shift=8.0, **kw):
    graph = common(**kw)
    graph["4"] = node("PowerShardWanLoader", checkpoint="wan2.2_ti2v_5B_fp16.safetensors", config=["2", 0],
                      keep_in_memory=True, options=["3", 0])
    graph["8"] = node("VAELoader", vae_name="wan2.2_vae.safetensors")
    graph["10"] = node("ModelSamplingSD3", model=["4", 0], shift=shift)
    latent = dict(vae=["8", 0], width=width, height=height, length=length, batch_size=1)
    if image:
        graph["9"] = node("LoadImage", image="example.png")
        latent["start_image"] = ["9", 0]
    graph["12"] = node("Wan22ImageToVideoLatent", **latent)
    graph["13"] = node("KSampler", model=["10", 0], positive=["6", 0], negative=["7", 0], latent_image=["12", 0],
                       seed=42, steps=steps, cfg=cfg, sampler_name="uni_pc", scheduler="simple", denoise=1.0)
    return tail(graph, ["13", 0], ["8", 0], 24, name)


def ui(graph):
    nodes, byid, links = [], {}, []
    for index, (key, n) in enumerate(graph.items()):
        sockets, outputs, widgets = SPECS[n["class_type"]]
        values = []
        for w in widgets:
            if w in n["inputs"]:
                values.append(n["inputs"][w])
                if SEED_CONTROL.get(n["class_type"]) == w:
                    values.append("fixed")
        if n["class_type"] == "LoadImage":
            values.append("image")
        obj = dict(id=int(key), type=n["class_type"], pos=[(index // 6) * 420, (index % 6) * 250], size=[380, 220], flags={},
                   order=index, mode=0, inputs=[dict(name=s, type=t, link=None) for s, t in sockets if s in n["inputs"]],
                   outputs=[dict(name=t, type=t, links=[], slot_index=i) for i, t in enumerate(outputs)],
                   properties={"Node name for S&R": n["class_type"], "powershard_schema": 7}, widgets_values=values)
        nodes.append(obj)
        byid[key] = obj
    for key, n in graph.items():
        for idx, inp in enumerate(byid[key]["inputs"]):
            source, slot = n["inputs"][inp["name"]]
            lid = len(links) + 1
            links.append([lid, int(source), slot, int(key), idx, inp["type"]])
            inp["link"] = lid
            byid[source]["outputs"][slot]["links"].append(lid)
    return dict(last_node_id=max(x["id"] for x in nodes), last_link_id=len(links), nodes=nodes, links=links, groups=[],
                config={}, extra={"ds": {"scale": .7, "offset": [30, 30]}}, version=.4)


def graphs():
    t2v = ("wan2.2_t2v_high_noise_14B_fp16.safetensors", "wan2.2_t2v_low_noise_14B_fp16.safetensors")
    i2v = ("wan2.2_i2v_high_noise_14B_fp16.safetensors", "wan2.2_i2v_low_noise_14B_fp16.safetensors")
    yield "wan22_t2v_a14b_480p_gpu_ulysses", moe_two_samplers("wan22_t2v_a14b_480p", *t2v)
    yield "wan22_t2v_a14b_720p_cpu_ulysses", moe_two_samplers(
        "wan22_t2v_a14b_720p", *t2v, width=1280, height=720, placement="cpu", residency="both", batch_chunk=1, prefetch=1)
    yield "wan22_t2v_a14b_lightning_4step", moe_two_samplers(
        "wan22_t2v_a14b_lightning", *t2v, steps=4, split=2, cfg=1.0, shift=5.0,
        loras=(("wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "high_noise"),
               ("wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", "low_noise")))
    yield "wan22_i2v_a14b_480p_gpu_ulysses", moe_two_samplers("wan22_i2v_a14b_480p", *i2v, boundary=0.9, image=True)
    yield "wan22_t2v_a14b_moe_single_sampler", moe_single_sampler("wan22_t2v_a14b_moe", *t2v)
    yield "wan22_ti2v_5b_720p_i2v", ti2v_5b("wan22_ti2v_5b_i2v")
    yield "wan22_ti2v_5b_720p_t2v", ti2v_5b("wan22_ti2v_5b_t2v", image=False)


# ---- Варианты (только API JSON: у LoadVideo/LoadAudio upload-виджеты зависят от версии frontend;
# ComfyUI открывает API JSON перетаскиванием) ----
def single(graph, checkpoint, placement="cpu"):
    graph["1"]["inputs"]["weight_placement"] = placement
    graph["4"] = node("PowerShardWanLoader", checkpoint=checkpoint, config=["2", 0], keep_in_memory=True, options=["3", 0])
    graph["10"] = node("ModelSamplingSD3", model=["4", 0], shift=8.0)
    return graph


def sampler(graph, positive, negative, latent, steps=20, cfg=5.0, model=("10", 0)):
    graph["13"] = node("KSampler", model=list(model), positive=positive, negative=negative, latent_image=latent,
                       seed=42, steps=steps, cfg=cfg, sampler_name="uni_pc", scheduler="simple", denoise=1.0)
    return ["13", 0]


def vace_14b(name):
    graph = single(common(), "wan2.1_vace_14B_fp16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["30"] = node("LoadVideo", file="control_depth.mp4")
    graph["31"] = node("GetVideoComponents", video=["30", 0])
    graph["12"] = node("WanVaceToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], control_video=["31", 0],
                       width=832, height=480, length=81, batch_size=1, strength=1.0)
    samples = sampler(graph, ["12", 0], ["12", 1], ["12", 2], cfg=4.0)
    graph["14"] = node("TrimVideoLatent", samples=samples, trim_amount=["12", 3])
    return tail(graph, ["14", 0], ["8", 0], 16, name)


def s2v_14b(name):
    graph = single(common(), "wan2.2_s2v_14B_bf16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="example.png")
    graph["30"] = node("LoadAudio", audio="speech.wav")
    graph["31"] = node("AudioEncoderLoader", audio_encoder_name="wav2vec2_large_english_fp16.safetensors")
    graph["32"] = node("AudioEncoderEncode", audio_encoder=["31", 0], audio=["30", 0])
    graph["12"] = node("WanSoundImageToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0],
                       audio_encoder_output=["32", 0], ref_image=["9", 0], width=832, height=480, length=77, batch_size=1)
    samples = sampler(graph, ["12", 0], ["12", 1], ["12", 2], cfg=4.5)
    graph = tail(graph, samples, ["8", 0], 16, name)
    graph["22"]["inputs"]["audio"] = ["30", 0]
    return graph


def animate_14b(name):
    graph = single(common(), "wan2.2_animate_14B_bf16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="character.png")
    graph["33"] = node("CLIPVisionLoader", clip_name="clip_vision_h.safetensors")
    graph["34"] = node("CLIPVisionEncode", clip_vision=["33", 0], image=["9", 0], crop="none")
    graph["30"] = node("LoadVideo", file="pose.mp4")
    graph["31"] = node("GetVideoComponents", video=["30", 0])
    graph["35"] = node("LoadVideo", file="face.mp4")
    graph["36"] = node("GetVideoComponents", video=["35", 0])
    graph["12"] = node("WanAnimateToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], clip_vision_output=["34", 0],
                       reference_image=["9", 0], pose_video=["31", 0], face_video=["36", 0], width=832, height=480,
                       length=77, batch_size=1, continue_motion_max_frames=5, video_frame_offset=0)
    samples = sampler(graph, ["12", 0], ["12", 1], ["12", 2], steps=6, cfg=1.0)
    graph["14"] = node("TrimVideoLatent", samples=samples, trim_amount=["12", 3])
    return tail(graph, ["14", 0], ["8", 0], 16, name)


def uni3c_i2v(name):
    graph = single(common(), "wan2.1_i2v_480p_14B_fp16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="example.png")
    graph["33"] = node("CLIPVisionLoader", clip_name="clip_vision_h.safetensors")
    graph["34"] = node("CLIPVisionEncode", clip_vision=["33", 0], image=["9", 0], crop="none")
    graph["12"] = node("WanImageToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], start_image=["9", 0],
                       clip_vision_output=["34", 0], width=832, height=480, length=81, batch_size=1)
    graph["30"] = node("LoadVideo", file="render_pointcloud.mp4")
    graph["31"] = node("GetVideoComponents", video=["30", 0])
    graph["15"] = node("PowerShardWanUni3C", model=["10", 0], controlnet="uni3c_controlnet.safetensors", vae=["8", 0],
                       render_video=["31", 0], latent=["12", 2], strength=1.0, start_percent=0.0, end_percent=1.0)
    samples = sampler(graph, ["12", 0], ["12", 1], ["12", 2], model=("15", 0))
    return tail(graph, samples, ["8", 0], 16, name)


def clip_vision(graph, image):
    graph["33"] = node("CLIPVisionLoader", clip_name="clip_vision_h.safetensors")
    graph["34"] = node("CLIPVisionEncode", clip_vision=["33", 0], image=image, crop="none")
    return ["34", 0]


def audio(graph, file, encoder):
    graph["30"] = node("LoadAudio", audio=file)
    graph["31"] = node("AudioEncoderLoader", audio_encoder_name=encoder)
    graph["32"] = node("AudioEncoderEncode", audio_encoder=["31", 0], audio=["30", 0])
    return ["32", 0]


def humo_17b(name):
    graph = single(common(), "humo_17B_fp16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="person.png")
    encoded = audio(graph, "speech.wav", "whisper_large_v3_fp16.safetensors")
    graph["12"] = node("WanHuMoImageToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], audio_encoder_output=encoded,
                       ref_image=["9", 0], width=832, height=480, length=97, batch_size=1)
    samples = sampler(graph, ["12", 0], ["12", 1], ["12", 2], steps=30, cfg=5.0)
    graph = tail(graph, samples, ["8", 0], 25, name)
    graph["22"]["inputs"]["audio"] = ["30", 0]
    return graph


def scail_14b(name):
    graph = single(common(), "wan2.1_scail_preview_14B_fp16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="character.png")
    graph["35"] = node("LoadVideo", file="pose_render.mp4")
    graph["36"] = node("GetVideoComponents", video=["35", 0])
    graph["12"] = node("WanSCAILToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], width=512, height=896,
                       length=81, batch_size=1, pose_video=["36", 0], reference_image=["9", 0],
                       clip_vision_output=clip_vision(graph, ["9", 0]), replacement_mode=False, pose_strength=1.0,
                       pose_start=0.0, pose_end=1.0, video_frame_offset=0, previous_frame_count=5)
    samples = sampler(graph, ["12", 0], ["12", 1], ["12", 2], cfg=4.0)
    return tail(graph, samples, ["8", 0], 16, name)


def wandancer_14b(name):
    graph = single(common(), "wan2.2_wandancer_14B_fp16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="dancer.png")
    graph["30"] = node("LoadAudio", audio="music.wav")
    graph["32"] = node("WanDancerEncodeAudio", audio=["30", 0], video_frames=149, audio_inject_scale=1.0)
    encoded = clip_vision(graph, ["9", 0])
    graph["12"] = node("WanDancerVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], width=480, height=832,
                       length=149, clip_vision_output=encoded, clip_vision_output_ref=encoded, start_image=["9", 0],
                       audio_encoder_output=["32", 0])
    samples = sampler(graph, ["12", 0], ["12", 1], ["12", 2], steps=30, cfg=4.0)
    graph = tail(graph, samples, ["8", 0], 30, name)
    graph["22"]["inputs"]["audio"] = ["30", 0]
    return graph


def animate2_14b(name):
    graph = common()
    graph["3"]["inputs"]["model_type"] = "animate2"  # checkpoint формы Wan2.1 I2V: тип задаётся явно
    graph = single(graph, "wan_animate2_14B_fp16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="character.png")
    graph["35"] = node("LoadVideo", file="driving.mp4")
    graph["36"] = node("GetVideoComponents", video=["35", 0])
    graph["12"] = node("WanAnimate2ToVideo", positive=["6", 0], negative=["7", 0], vae=["8", 0], width=832, height=480,
                       length=81, batch_size=1, reference_image=["9", 0], pose_video=["36", 0],
                       clip_vision_output=clip_vision(graph, ["9", 0]), video_frame_offset=0, pose_strength=1.0,
                       pose_start_percent=0.0, pose_end_percent=1.0, reference_image_strength=1.0)
    graph["15"] = node("PowerShardWanAnimate2Cache", model=["10", 0], dtype="fp16")
    # CLIP vision / VAE выгружаются с cuda:0 (это и rank 0 workers) до первого шага sampling.
    graph["16"] = node("PowerShardFreeVRAM", positive=["12", 0], negative=["12", 1], mode="selected",
                       clip_vision=["33", 0], vae=["8", 0], latent=["12", 2])
    samples = sampler(graph, ["16", 0], ["16", 1], ["16", 2], cfg=1.0, steps=8, model=("15", 0))
    graph["14"] = node("TrimVideoLatent", samples=samples, trim_amount=["12", 3])
    return tail(graph, ["14", 0], ["8", 0], 16, name)


def infinitetalk_14b(name, speakers=1):
    graph = single(common(), "wan2.1_i2v_480p_14B_fp16.safetensors")
    graph["8"] = node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    graph["9"] = node("LoadImage", image="speakers.png" if speakers == 2 else "person.png")
    encoded = audio(graph, "speech.wav", "wav2vec2-chinese-base_fp16.safetensors")
    inputs = dict(mode="single_speaker", model=["10", 0], model_patch="wan2.1_infiniteTalk_single_fp16.safetensors",
                  positive=["6", 0], negative=["7", 0], vae=["8", 0], width=832, height=480, length=81,
                  audio_encoder_output_1=encoded, motion_frame_count=9, audio_scale=1.0,
                  clip_vision_output=clip_vision(graph, ["9", 0]), start_image=["9", 0])
    if speakers == 2:
        graph["37"] = node("LoadAudio", audio="speech_2.wav")
        graph["38"] = node("AudioEncoderEncode", audio_encoder=["31", 0], audio=["37", 0])
        graph["39"] = node("LoadImage", image="mask_speaker_1.png")
        graph["40"] = node("LoadImage", image="mask_speaker_2.png")
        inputs.update(mode="two_speakers", model_patch="wan2.1_infiniteTalk_multi_fp16.safetensors",
                      audio_encoder_output_2=["38", 0], mask_1=["39", 1], mask_2=["40", 1])
    graph["15"] = node("PowerShardWanInfiniteTalk", **inputs)
    samples = sampler(graph, ["15", 1], ["15", 2], ["15", 3], steps=6, cfg=1.0, model=("15", 0))
    graph = tail(graph, samples, ["8", 0], 25, name)
    graph["22"]["inputs"]["audio"] = ["30", 0]
    return graph


def variant_graphs():
    yield "wan21_vace_14b_control", vace_14b("wan21_vace_14b")
    yield "wan22_s2v_14b", s2v_14b("wan22_s2v_14b")
    yield "wan22_animate_14b", animate_14b("wan22_animate_14b")
    yield "wan21_i2v_14b_uni3c_controlnet", uni3c_i2v("wan21_i2v_uni3c")
    yield "wan21_humo_17b", humo_17b("wan21_humo")
    yield "wan21_scail_14b", scail_14b("wan21_scail")
    yield "wan22_wandancer_14b", wandancer_14b("wan22_wandancer")
    yield "wan_animate2_14b", animate2_14b("wan_animate2")
    yield "wan21_infinitetalk_14b", infinitetalk_14b("wan21_infinitetalk")
    yield "wan21_infinitetalk_14b_two_speakers", infinitetalk_14b("wan21_infinitetalk_duo", speakers=2)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, graph in graphs():
        for suffix, data in (("api", graph), ("ui", ui(graph))):
            (OUT / f"{name}.{suffix}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(name)
    for name, graph in variant_graphs():
        (OUT / f"{name}.api.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False))
        print(name, "(API)")
