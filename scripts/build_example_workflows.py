"""Сборка example_workflows/*.json (UI-формат ComfyUI: перетащите файл в окно ComfyUI).

  python scripts/build_example_workflows.py          # перезаписать файлы
  python scripts/build_example_workflows.py --check  # только сравнить (используется тестом)

Порядок виджетов PowerShard-нод берётся из их INPUT_TYPES (без torch), родных нод ComfyUI — из таблицы
NATIVE ниже (сверено с ComfyUI 0.x, октябрь 2026). Имена файлов моделей — примеры: выберите свои в нодах.
"""
import argparse
import json
import sys
import types
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "example_workflows"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# Родные ноды: (входы по порядку: имя -> тип сокета или "W" для виджета, выходы)
NATIVE = {
    "CLIPTextEncode": ([("text", "W"), ("clip", "CLIP")], ["CONDITIONING"]),
    "VAELoader": ([("vae_name", "W")], ["VAE"]),
    "ModelSamplingSD3": ([("model", "MODEL"), ("shift", "W")], ["MODEL"]),
    "EmptyHunyuanLatentVideo": ([("width", "W"), ("height", "W"), ("length", "W"), ("batch_size", "W")], ["LATENT"]),
    "KSampler": ([("model", "MODEL"), ("seed", "W"), ("steps", "W"), ("cfg", "W"), ("sampler_name", "W"),
                  ("scheduler", "W"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                  ("latent_image", "LATENT"), ("denoise", "W")], ["LATENT"]),
    "VAEDecodeTiled": ([("samples", "LATENT"), ("vae", "VAE"), ("tile_size", "W"), ("overlap", "W"),
                        ("temporal_size", "W"), ("temporal_overlap", "W")], ["IMAGE"]),
    "CreateVideo": ([("images", "IMAGE"), ("fps", "W"), ("audio", "AUDIO"), ("bit_depth", "W"), ("color_space", "W"),
                     ("codec", "W")], ["VIDEO"]),
    "SaveVideo": ([("video", "VIDEO"), ("filename_prefix", "W"), ("format", "W"), ("codec", "W")], ["VIDEO"]),
    "LoadImage": ([("image", "W"), ("upload", "W")], ["IMAGE", "MASK"]),
    "WanImageToVideo": ([("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"), ("width", "W"),
                         ("height", "W"), ("length", "W"), ("batch_size", "W"), ("clip_vision_output", "CLIP_VISION_OUTPUT"),
                         ("start_image", "IMAGE"), ("ref_pad_image", "IMAGE")], ["CONDITIONING", "CONDITIONING", "LATENT"]),
    "Wan22ImageToVideoLatent": ([("vae", "VAE"), ("width", "W"), ("height", "W"), ("length", "W"), ("batch_size", "W"),
                                 ("start_image", "IMAGE")], ["LATENT"]),
    "LTXVConditioning": ([("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("frame_rate", "W")],
                         ["CONDITIONING", "CONDITIONING"]),
    "EmptyLTXVLatentVideo": ([("width", "W"), ("height", "W"), ("length", "W"), ("batch_size", "W")], ["LATENT"]),
    "LTXVEmptyLatentAudio": ([("frames_number", "W"), ("frame_rate", "W"), ("batch_size", "W"), ("audio_vae", "VAE")],
                             ["LATENT"]),
    "KSamplerSelect": ([("sampler_name", "W")], ["SAMPLER"]),
    "RandomNoise": ([("noise_seed", "W")], ["NOISE"]),
    "LTXVSpatioTemporalGuidance": ([("model", "MODEL"), ("scale", "W"), ("blocks", "W"), ("start_percent", "W"),
                                    ("end_percent", "W")], ["MODEL"]),
    "LTXVModalityGuidance": ([("model", "MODEL"), ("modality_scale", "W"), ("start_percent", "W"), ("end_percent", "W")],
                             ["MODEL"]),
    "LTXVConcatAVLatent": ([("video_latent", "LATENT"), ("audio_latent", "LATENT")], ["LATENT"]),
    "LTXVScheduler": ([("steps", "W"), ("max_shift", "W"), ("base_shift", "W"), ("stretch", "W"), ("terminal", "W"),
                       ("latent", "LATENT")], ["SIGMAS"]),
    "LTXVDualCFGGuider": ([("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                           ("video_cfg", "W"), ("audio_cfg", "W")], ["GUIDER"]),
    "SamplerCustomAdvanced": ([("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                               ("latent_image", "LATENT")], ["LATENT", "LATENT"]),
    "LTXVSeparateAVLatent": ([("av_latent", "LATENT")], ["LATENT", "LATENT"]),
    "LTXVAudioVAEDecode": ([("samples", "LATENT"), ("audio_vae", "VAE")], ["AUDIO"]),
    "LTXVPreprocess": ([("image", "IMAGE"), ("img_compression", "W")], ["IMAGE"]),
    "LTXVImgToVideoInplace": ([("vae", "VAE"), ("image", "IMAGE"), ("latent", "LATENT"), ("strength", "W"),
                               ("bypass", "W")], ["LATENT"]),
    "MiniMaxH3ImageToVideo": ([("clip", "CLIP"), ("vae", "VAE"), ("prompt", "W"), ("width", "W"), ("height", "W"),
                               ("length", "W"), ("first_frame", "IMAGE"), ("last_frame", "IMAGE")],
                              ["CONDITIONING", "LATENT"]),
    "BasicGuider": ([("model", "MODEL"), ("conditioning", "CONDITIONING")], ["GUIDER"]),
    "BasicScheduler": ([("model", "MODEL"), ("scheduler", "W"), ("steps", "W"), ("denoise", "W")], ["SIGMAS"]),
    "VAEDecodeAudio": ([("samples", "LATENT"), ("vae", "VAE")], ["AUDIO"]),
    "EmptyLatentImage": ([("width", "W"), ("height", "W"), ("batch_size", "W")], ["LATENT"]),
    "EmptySD3LatentImage": ([("width", "W"), ("height", "W"), ("batch_size", "W")], ["LATENT"]),
    "EmptyFlux2LatentImage": ([("width", "W"), ("height", "W"), ("batch_size", "W")], ["LATENT"]),
    "FluxGuidance": ([("conditioning", "CONDITIONING"), ("guidance", "W")], ["CONDITIONING"]),
    "ModelSamplingAuraFlow": ([("model", "MODEL"), ("shift", "W"), ("sampling", "W")], ["MODEL"]),
    "VAEDecode": ([("samples", "LATENT"), ("vae", "VAE")], ["IMAGE"]),
    "SaveImage": ([("images", "IMAGE"), ("filename_prefix", "W")], []),
    "Note": ([("text", "W")], []),
}
SEED_CONTROL = {"KSampler": "seed", "RandomNoise": "noise_seed"}
DEFAULTS = {"PowerShardTextEncoder": dict(text_encoder_2="none", text_encoder_3="none", text_encoder_4="none",
                                          ltx_checkpoint="none"),
            "ModelSamplingAuraFlow": dict(sampling="flow"),
            "CreateVideo": dict(bit_depth="auto", color_space="sRGB", codec="none"),
            "SaveVideo": dict(format="auto", codec="auto"), "LoadImage": dict(upload="image")}


def powershard_specs():
    """INPUT_TYPES PowerShard-нод без torch и ComfyUI (folder_paths подменяется)."""
    fake = types.ModuleType("folder_paths")
    fake.get_filename_list = lambda folder: []
    fake.models_dir = ""
    fake.folder_names_and_paths = {}
    saved = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = fake
    sys.path.insert(0, str(ROOT))
    try:
        import powershard.veda_nodes as veda_nodes
        register_folder, veda_nodes.register_folder = veda_nodes.register_folder, lambda: None
        from powershard.nodes import NODE_CLASS_MAPPINGS as a
        from powershard.wan_nodes import WAN_NODE_CLASS_MAPPINGS as b
        from powershard.ltx_nodes import LTX_NODE_CLASS_MAPPINGS as c
        from powershard.common_nodes import COMMON_NODE_CLASS_MAPPINGS as d
        specs = {}
        for name, cls in {**a, **b, **c, **d, **veda_nodes.VEDA_NODE_CLASS_MAPPINGS}.items():
            fields = cls.INPUT_TYPES()
            inputs = []
            for group in ("required", "optional"):
                for key, value in fields.get(group, {}).items():
                    kind = value[0]
                    widget = isinstance(kind, (list, tuple)) or kind in WIDGET_TYPES
                    inputs.append((key, "W" if widget else kind))
            specs[name] = (inputs, list(cls.RETURN_TYPES))
        veda_nodes.register_folder = register_folder
        return specs
    finally:
        if saved is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = saved


SPECS = dict(NATIVE)
SPECS.update(powershard_specs())


# ------------------------------------------------------------------ API-граф -> UI
class Graph:
    def __init__(self):
        self.nodes = OrderedDict()
        self.titles = {}

    def add(self, class_type, title=None, **inputs):
        key = str(len(self.nodes) + 1)
        self.nodes[key] = dict(class_type=class_type, inputs=inputs)
        if title:
            self.titles[key] = title
        return key

    def ui(self, notes=()):
        depth = {}

        def level(key):
            if key not in depth:
                parents = [v[0] for v in self.nodes[key]["inputs"].values() if isinstance(v, list)]
                depth[key] = 1 + max((level(p) for p in parents), default=-1)
            return depth[key]
        for key in self.nodes:
            level(key)
        rows, nodes, links, byid = {}, [], [], {}
        for order, (key, node) in enumerate(self.nodes.items()):
            cls = node["class_type"]
            spec_inputs, outputs = SPECS[cls]
            values, sockets = [], []
            given = dict(DEFAULTS.get(cls, {}), **node["inputs"])
            for name, kind in spec_inputs:
                if kind == "W":
                    if name not in given:
                        raise KeyError(f"{cls}: не задан виджет {name}")
                    values.append(given[name])
                    if SEED_CONTROL.get(cls) == name:
                        values.append("fixed")
                else:
                    sockets.append(dict(name=name, type=kind, link=None))
            col = depth[key]
            y = rows.get(col, 0)
            height = 46 + 26 * max(len(values), 1) + 22 * len(sockets)
            if cls in ("CLIPTextEncode", "MiniMaxH3ImageToVideo"):
                height += 120
            rows[col] = y + height + 40
            obj = dict(id=int(key), type=cls, pos=[40 + col * 380, 40 + y], size=[340, height], flags={}, order=order,
                       mode=0, inputs=sockets,
                       outputs=[dict(name=t, type=t, links=[], slot_index=i) for i, t in enumerate(outputs)],
                       properties={"Node name for S&R": cls}, widgets_values=values)
            if key in self.titles:
                obj["title"] = self.titles[key]
            nodes.append(obj)
            byid[key] = obj
        for key, node in self.nodes.items():
            for slot, socket in enumerate(byid[key]["inputs"]):
                source = node["inputs"].get(socket["name"])
                if source is None:
                    continue
                link_id = len(links) + 1
                links.append([link_id, int(source[0]), source[1], int(key), slot, socket["type"]])
                socket["link"] = link_id
                byid[source[0]]["outputs"][source[1]]["links"].append(link_id)
        next_id = len(nodes) + 1
        for i, (text, width) in enumerate(notes):
            nodes.append(dict(id=next_id + i, type="Note", pos=[40, -260 - 220 * i], size=[width, 200], flags={},
                              order=len(nodes), mode=0, inputs=[], outputs=[], properties={"text": ""},
                              color="#432", bgcolor="#653", widgets_values=[text]))
        return dict(last_node_id=max(n["id"] for n in nodes), last_link_id=len(links), nodes=nodes, links=links,
                    groups=[], config={}, extra={"ds": {"scale": 0.65, "offset": [40, 320]}}, version=0.4)


# ------------------------------------------------------------------ общие куски
NEG_WAN = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
           "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
           "杂乱的背景，三条腿，背景人很多，倒着走")
PROMPT_WAN = ("A slow cinematic orbit around a lighthouse on a rocky shore at sunset, waves crashing on the rocks, "
              "warm golden light.")
PROMPT_LTX = ("A lighthouse keeper climbs the spiral stairs at dusk, his lantern swinging; waves crash on the rocks "
              "outside and the wind howls through the window. He says quietly: \"Another storm tonight.\"")
NEG_LTX = ("blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles, distorted, "
           "deformed, disfigured, bad anatomy, static, jitter")
PROMPT_H3 = "Waves roll onto a rocky shore at sunset. The camera is static. The sound of the sea and seagulls."

COMMON_NOTE = ("PowerShard: модель шардируется по всем GPU из «GPU / память / attention» (gpu_ids=all, sequence_mode="
               "ulysses). Имена файлов — примеры, выберите свои. Release перед VAE освобождает GPU workers.\n"
               "EN: the model is sharded across the GPUs selected in the config node. File names are examples - pick "
               "your own. Release frees worker VRAM before VAE decode.")


def config(g):
    return g.add("PowerShardConfig", gpu_ids="all", weight_placement="gpu", precision="fp16", attention_backend="auto",
                 sequence_mode="ulysses")


def decode_and_save(g, samples, vae, fps, prefix, audio=None):
    released = g.add("PowerShardRelease", samples=samples, preserve_qwen_cpu_shards=True,
                     clear_conditioning_cache=False, preserve_h3_cpu_shards=True)
    if audio is None:
        image = g.add("VAEDecodeTiled", samples=[released, 0], vae=vae, tile_size=512, overlap=64, temporal_size=64,
                      temporal_overlap=8)
        video = g.add("CreateVideo", images=[image, 0], fps=float(fps))
    else:
        separated = g.add("LTXVSeparateAVLatent", av_latent=[released, 0])
        image = g.add("VAEDecodeTiled", samples=[separated, 0], vae=vae, tile_size=512, overlap=64, temporal_size=64,
                      temporal_overlap=8)
        sound = audio(g, [separated, 1])
        video = g.add("CreateVideo", images=[image, 0], audio=[sound, 0], fps=float(fps))
    g.add("SaveVideo", video=[video, 0], filename_prefix=f"video/{prefix}")


def wan_text(g, cfg):
    te = g.add("PowerShardTextEncoder", text_encoder="umt5_xxl_fp16.safetensors", type="wan_umt5", placement="workers",
               after_encode="release", config=[cfg, 0], ltx_checkpoint="none")
    pos = g.add("CLIPTextEncode", title="Positive", clip=[te, 0], text=PROMPT_WAN)
    neg = g.add("CLIPTextEncode", title="Negative", clip=[te, 0], text=NEG_WAN)
    return pos, neg


def wan_options(g):
    return g.add("PowerShardWanOptions", fp16_safe=False, mlp_chunk_mode="auto", mlp_chunk_tokens=4096,
                 moe_residency="swap", batch_chunk=0, model_type="auto")


# ------------------------------------------------------------------ workflows
def wan22_a14b(image):
    g = Graph()
    cfg = config(g)
    opts = wan_options(g)
    kind = "i2v" if image else "t2v"
    loader = g.add("PowerShardWan22MoELoader", high_noise_checkpoint=f"wan2.2_{kind}_high_noise_14B_fp16.safetensors",
                   low_noise_checkpoint=f"wan2.2_{kind}_low_noise_14B_fp16.safetensors", config=[cfg, 0], boundary=0.0,
                   keep_in_memory=True, options=[opts, 0])
    pos, neg = wan_text(g, cfg)
    vae = g.add("VAELoader", vae_name="wan_2.1_vae.safetensors")
    model = g.add("ModelSamplingSD3", model=[loader, 2], shift=8.0)
    if image:
        picture = g.add("LoadImage", image="example.png")
        cond = g.add("WanImageToVideo", positive=[pos, 0], negative=[neg, 0], vae=[vae, 0], start_image=[picture, 0],
                     width=832, height=480, length=81, batch_size=1)
        positive, negative, latent = [cond, 0], [cond, 1], [cond, 2]
    else:
        empty = g.add("EmptyHunyuanLatentVideo", width=832, height=480, length=81, batch_size=1)
        positive, negative, latent = [pos, 0], [neg, 0], [empty, 0]
    samples = g.add("KSampler", model=[model, 0], positive=positive, negative=negative, latent_image=latent, seed=42,
                    steps=20, cfg=3.5, sampler_name="euler", scheduler="simple", denoise=1.0)
    decode_and_save(g, [samples, 0], [vae, 0], 16, f"wan22_{kind}_a14b")
    note = ("Wan 2.2 A14B " + ("I2V" if image else "T2V") + ": high+low эксперты в одном пуле workers; выход moe "
            "переключает эксперта сам (boundary=0 — авто: T2V 0.875, I2V 0.900), поэтому один KSampler.\n"
            "EN: both experts share one worker pool; the moe output switches experts automatically, so a single "
            "KSampler is enough.")
    return g.ui([(COMMON_NOTE, 760), (note, 760)])


def wan22_ti2v_5b():
    g = Graph()
    cfg = config(g)
    opts = wan_options(g)
    loader = g.add("PowerShardWanLoader", checkpoint="wan2.2_ti2v_5B_fp16.safetensors", config=[cfg, 0],
                   keep_in_memory=True, options=[opts, 0])
    pos, neg = wan_text(g, cfg)
    vae = g.add("VAELoader", vae_name="wan2.2_vae.safetensors")
    model = g.add("ModelSamplingSD3", model=[loader, 0], shift=8.0)
    picture = g.add("LoadImage", image="example.png")
    latent = g.add("Wan22ImageToVideoLatent", vae=[vae, 0], start_image=[picture, 0], width=1280, height=704, length=121,
                   batch_size=1)
    samples = g.add("KSampler", model=[model, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                    seed=42, steps=30, cfg=5.0, sampler_name="uni_pc", scheduler="simple", denoise=1.0)
    decode_and_save(g, [samples, 0], [vae, 0], 24, "wan22_ti2v_5b")
    note = ("Wan 2.2 TI2V 5B, 720p: с подключённой картинкой — image-to-video, без неё (отключите связь start_image) — "
            "text-to-video.\nEN: image-to-video with the image connected; disconnect start_image for text-to-video.")
    return g.ui([(COMMON_NOTE, 760), (note, 760)])


LTX_CKPT = "checkpoints/ltx-2.3-22b-dev.safetensors"


def ltx2(image):
    g = Graph()
    cfg = config(g)
    opts = g.add("PowerShardLTXOptions", fp16_safe=False, mlp_chunk_mode="auto", mlp_chunk_tokens=4096, batch_chunk=0,
                 attention_chunk=8192)
    loader = g.add("PowerShardLTXLoader", checkpoint=LTX_CKPT, config=[cfg, 0], keep_in_memory=True, options=[opts, 0])
    te = g.add("PowerShardTextEncoder", text_encoder="gemma_3_12B_it.safetensors", type="ltx_gemma", placement="workers",
               after_encode="release", config=[cfg, 0], ltx_checkpoint=LTX_CKPT)
    pos = g.add("CLIPTextEncode", title="Positive", clip=[te, 0], text=PROMPT_LTX)
    neg = g.add("CLIPTextEncode", title="Negative", clip=[te, 0], text=NEG_LTX)
    cond = g.add("LTXVConditioning", positive=[pos, 0], negative=[neg, 0], frame_rate=24.0)
    vaes = g.add("PowerShardLTXVAELoader", ckpt_name=LTX_CKPT)
    video = g.add("EmptyLTXVLatentVideo", width=768, height=512, length=121, batch_size=1)
    if image:
        picture = g.add("LoadImage", image="example.png")
        prepared = g.add("LTXVPreprocess", image=[picture, 0], img_compression=35)
        video = g.add("LTXVImgToVideoInplace", vae=[vaes, 0], image=[prepared, 0], latent=[video, 0], strength=1.0,
                      bypass=False)
    audio = g.add("LTXVEmptyLatentAudio", frames_number=121, frame_rate=24.0, batch_size=1, audio_vae=[vaes, 1])
    av = g.add("LTXVConcatAVLatent", video_latent=[video, 0], audio_latent=[audio, 0])
    stg = g.add("LTXVSpatioTemporalGuidance", model=[loader, 0], scale=1.0, blocks="29", start_percent=0.0,
                end_percent=1.0)
    modality = g.add("LTXVModalityGuidance", model=[stg, 0], modality_scale=3.0, start_percent=0.0, end_percent=1.0)
    sigmas = g.add("LTXVScheduler", steps=30, max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1, latent=[av, 0])
    guider = g.add("LTXVDualCFGGuider", model=[modality, 0], positive=[cond, 0], negative=[cond, 1], video_cfg=3.0,
                   audio_cfg=7.0)
    sampler = g.add("KSamplerSelect", sampler_name="euler")
    noise = g.add("RandomNoise", noise_seed=42)
    samples = g.add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0],
                    sigmas=[sigmas, 0], latent_image=[av, 0])
    decode_and_save(g, [samples, 0], [vaes, 0], 24, "ltx2_" + ("i2v" if image else "t2v"),
                    audio=lambda g, latent: g.add("LTXVAudioVAEDecode", samples=latent, audio_vae=[vaes, 1]))
    note = ("LTX-2.3 видео+звук" + (" из картинки" if image else "") + ": Gemma и DiT шардируются по GPU, VAE видео и звука "
            "читаются из того же checkpoint без весов DiT.\nEN: Gemma and the DiT are sharded across GPUs; both VAEs are "
            "read from the same checkpoint without loading the DiT weights.")
    return g.ui([(COMMON_NOTE, 760), (note, 760)])


def h3(veda):
    g = Graph()
    cfg = config(g)
    weights = g.add("PowerShardWeights", weight_format="as_file", compute="auto", weight_dtype="auto",
                    fp16_accumulation=False)
    loader = g.add("PowerShardH3QuantLoader", checkpoint="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                   config=[cfg, 0], keep_in_memory=True, weights=[weights, 0])
    model = loader
    if veda:
        model = g.add("PowerShardH3Veda", model=[model, 0],
                      predictor="minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors", generated_sparsity="32",
                      reference_sparsity="32", full_attention_layers="", full_attention_steps="", backend="auto")
    qwen = g.add("PowerShardH3QwenLoader", checkpoint="qwen3vl_32b_minimax_h3_int8_convrot.safetensors", config=[cfg, 0],
                 precision="int8_fp16", idle_policy="release", cache_mib=256)
    video_vae = g.add("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    audio_vae = g.add("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    cond = g.add("MiniMaxH3ImageToVideo", clip=[qwen, 0], vae=[video_vae, 0], prompt=PROMPT_H3, width=512, height=512,
                 length=25)
    guider = g.add("BasicGuider", model=[model, 0], conditioning=[cond, 0])
    noise = g.add("RandomNoise", noise_seed=44)
    sampler = g.add("KSamplerSelect", sampler_name="euler")
    sigmas = g.add("BasicScheduler", model=[model, 0], scheduler="simple", steps=8, denoise=1.0)
    samples = g.add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0],
                    sigmas=[sigmas, 0], latent_image=[cond, 1])
    decode_and_save(g, [samples, 0], [video_vae, 0], 24, "minimax_h3_t2va" + ("_veda" if veda else ""),
                    audio=lambda g, latent: g.add("VAEDecodeAudio", samples=latent, vae=[audio_vae, 0]))
    note = ("MiniMax H3 text-to-video+audio. Веса int8 хранятся как в файле (узел «формат весов»), на V100 GEMM "
            "считается через распаковку в fp16. Qwen3-VL 32B — отдельный пул workers, освобождается перед H3."
            + ("\nVeda: предиктор положите в models/veda (ссылка — во всплывающей подсказке и README). На V100 "
               "backend=auto выбирает torch-sdpa." if veda else "")
            + "\nEN: int8 weights are kept packed (weights node); Qwen3-VL runs in its own worker pool."
            + (" Put the Veda predictor into models/veda." if veda else ""))
    return g.ui([(COMMON_NOTE, 760), (note, 760)])


# ------------------------------------------------------------------ image
IMAGE_NOTE = ("PowerShard Image: модель шардируется по GPU (FSDP2), forward — родной код ComfyUI. parallel=auto: "
              "Flux / Qwen-Image / Z-Image / MOSS-Image — токены картинки по GPU (sequence), остальные (SDXL) — "
              "строки batch (cond/uncond) по GPU.\nEN: the model is sharded across GPUs; parallel=auto splits image "
              "tokens (sequence) for Flux / Qwen-Image / Z-Image / MOSS-Image and batch rows for other models (SDXL).")


def image_loader(g, cfg, checkpoint):
    return g.add("PowerShardImageLoader", checkpoint=checkpoint, config=[cfg, 0], parallel="auto", keep_in_memory=True)


def save_image(g, samples, vae, prefix):
    released = g.add("PowerShardRelease", samples=samples, preserve_qwen_cpu_shards=True,
                     clear_conditioning_cache=False, preserve_h3_cpu_shards=True)
    image = g.add("VAEDecode", samples=[released, 0], vae=vae)
    g.add("SaveImage", images=[image, 0], filename_prefix=f"image/{prefix}")


def text_pair(g, clip, positive, negative=""):
    pos = g.add("CLIPTextEncode", title="Positive", clip=clip, text=positive)
    neg = g.add("CLIPTextEncode", title="Negative", clip=clip, text=negative)
    return pos, neg


PROMPT_IMAGE = ("A cozy lighthouse keeper's room at dusk, warm lamp light, rain on the window, a cat sleeping on a "
                "stack of old maps, detailed, cinematic")


def sdxl():
    g = Graph()
    cfg = config(g)
    ckpt = "sd_xl_base_1.0.safetensors"
    model = image_loader(g, cfg, "checkpoints/" + ckpt)
    parts = g.add("PowerShardCheckpointCLIPVAE", ckpt_name=ckpt)
    pos, neg = text_pair(g, [parts, 0], PROMPT_IMAGE, "blurry, low quality, watermark, text")
    latent = g.add("EmptyLatentImage", width=1024, height=1024, batch_size=1)
    samples = g.add("KSampler", model=[model, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                    seed=42, steps=30, cfg=7.0, sampler_name="dpmpp_2m", scheduler="karras", denoise=1.0)
    save_image(g, [samples, 0], [parts, 1], "sdxl")
    note = ("SDXL: UNet по GPU, строки cond/uncond — на разных GPU (batch parallel). CLIP и VAE из того же "
            "checkpoint считаются на host без загрузки UNet.\nEN: the UNet runs on the workers (batch parallel); CLIP "
            "and VAE are read from the same checkpoint without the UNet.")
    return g.ui([(IMAGE_NOTE, 760), (note, 760)])


def flux1():
    g = Graph()
    cfg = config(g)
    model = image_loader(g, cfg, "diffusion_models/flux1-dev.safetensors")
    te = g.add("PowerShardTextEncoder", text_encoder="clip_l.safetensors", type="flux", placement="workers",
               after_encode="release", config=[cfg, 0], text_encoder_2="t5xxl_fp16.safetensors")
    pos, neg = text_pair(g, [te, 0], PROMPT_IMAGE)
    guided = g.add("FluxGuidance", conditioning=[pos, 0], guidance=3.5)
    latent = g.add("EmptySD3LatentImage", width=1024, height=1024, batch_size=1)
    samples = g.add("KSampler", model=[model, 0], positive=[guided, 0], negative=[neg, 0], latent_image=[latent, 0],
                    seed=42, steps=20, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    vae = g.add("VAELoader", vae_name="ae.safetensors")
    save_image(g, [samples, 0], [vae, 0], "flux1_dev")
    note = ("Flux.1 dev: 12B, ~4 GB весов на GPU при 6 GPU; токены картинки делятся между GPU. clip_l + t5xxl — "
            "тоже в workers.\nEN: 12B, about 4 GB of weights per GPU on 6 GPUs; image tokens are split across GPUs.")
    return g.ui([(IMAGE_NOTE, 760), (note, 760)])


def z_image():
    g = Graph()
    cfg = config(g)
    model = image_loader(g, cfg, "diffusion_models/z_image_turbo_bf16.safetensors")
    te = g.add("PowerShardTextEncoder", text_encoder="qwen_3_4b.safetensors", type="z_image", placement="workers",
               after_encode="release", config=[cfg, 0])
    pos, neg = text_pair(g, [te, 0], PROMPT_IMAGE)
    latent = g.add("EmptySD3LatentImage", width=1024, height=1024, batch_size=1)
    samples = g.add("KSampler", model=[model, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                    seed=42, steps=8, cfg=1.0, sampler_name="res_multistep", scheduler="simple", denoise=1.0)
    vae = g.add("VAELoader", vae_name="ae.safetensors")
    save_image(g, [samples, 0], [vae, 0], "z_image_turbo")
    note = ("Z-Image Turbo: 8 шагов, cfg 1. На V100 по умолчанию bf16 хранение + FP32 вычисление (у модели нет "
            "fp16); быстрее — weight_dtype=fp16 в узле формата весов (проверьте качество).\nEN: on V100 the default is "
            "bf16 storage with FP32 compute; weight_dtype=fp16 is faster but check the quality.")
    return g.ui([(IMAGE_NOTE, 760), (note, 760)])


def moss_image():
    g = Graph()
    cfg = config(g)
    model = image_loader(g, cfg, "diffusion_models/moss_image_sft_bf16.safetensors")
    te = g.add("PowerShardTextEncoder", text_encoder="qwen3.5_9b.safetensors", type="moss_image", placement="workers",
               after_encode="release", config=[cfg, 0])
    pos, neg = text_pair(g, [te, 0], PROMPT_IMAGE, "low quality, blurry, distorted")
    latent = g.add("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)
    samples = g.add("KSampler", model=[model, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                    seed=42, steps=50, cfg=3.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    vae = g.add("VAELoader", vae_name="flux2-vae.safetensors")
    save_image(g, [samples, 0], [vae, 0], "moss_image")
    note = ("MOSS-Image (5B, латенты Flux.2): 50 шагов, cfg 3, shift 5 (по умолчанию модели). Энкодер Qwen3.5-9B "
            "в workers.\nEN: 50 steps, cfg 3, the model's default shift 5; the Qwen3.5-9B encoder runs in workers.")
    return g.ui([(IMAGE_NOTE, 760), (note, 760)])


def qwen_image():
    g = Graph()
    cfg = config(g)
    model = image_loader(g, cfg, "diffusion_models/qwen_image_bf16.safetensors")
    shifted = g.add("ModelSamplingAuraFlow", model=[model, 0], shift=3.1)
    te = g.add("PowerShardTextEncoder", text_encoder="qwen_2.5_vl_7b.safetensors", type="qwen_image",
               placement="workers", after_encode="release", config=[cfg, 0])
    pos, neg = text_pair(g, [te, 0], PROMPT_IMAGE)
    latent = g.add("EmptySD3LatentImage", width=1328, height=1328, batch_size=1)
    samples = g.add("KSampler", model=[shifted, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                    seed=42, steps=20, cfg=2.5, sampler_name="euler", scheduler="simple", denoise=1.0)
    vae = g.add("VAELoader", vae_name="qwen_image_vae.safetensors")
    save_image(g, [samples, 0], [vae, 0], "qwen_image")
    note = ("Qwen-Image (20B): bf16 ≈ 7 GB весов на GPU при 6 GPU. На V100 — bf16 хранение + FP32 вычисление.\n"
            "EN: about 7 GB of bf16 weights per GPU on 6 GPUs; on V100 the compute is FP32.")
    return g.ui([(IMAGE_NOTE, 760), (note, 760)])


WORKFLOWS = {
    "wan22_t2v_a14b": lambda: wan22_a14b(False),
    "wan22_i2v_a14b": lambda: wan22_a14b(True),
    "wan22_ti2v_5b": wan22_ti2v_5b,
    "ltx2_t2v_audio": lambda: ltx2(False),
    "ltx2_i2v_audio": lambda: ltx2(True),
    "minimax_h3_t2va": lambda: h3(False),
    "minimax_h3_t2va_veda": lambda: h3(True),
    "sdxl_t2i": sdxl,
    "flux1_dev_t2i": flux1,
    "z_image_turbo_t2i": z_image,
    "moss_image_t2i": moss_image,
    "qwen_image_t2i": qwen_image,
}


def render(name):
    return json.dumps(WORKFLOWS[name](), indent=1, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    stale = []
    for name in WORKFLOWS:
        path = OUT / f"{name}.json"
        text = render(name)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
            print(path.relative_to(ROOT))
    if stale:
        sys.exit("устарели: " + ", ".join(stale) + " — запустите scripts/build_example_workflows.py")


if __name__ == "__main__":
    main()
