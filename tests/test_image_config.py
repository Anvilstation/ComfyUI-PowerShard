"""Image-модели: раскладка sequence parallel, выбор режима, ноды. Без torch/CUDA."""
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def brute_local_positions(segments, world, rank):
    from powershard.image_layout import split_sizes
    out, start = [], 0
    for n, split in segments:
        if split:
            sizes = split_sizes(n, world)
            a = start + sum(sizes[:rank])
            out += list(range(a, a + sizes[rank]))
        else:
            out += list(range(start, start + n))
        start += n
    return out


def test_split_sizes_cover_and_balance():
    from powershard.image_layout import split_sizes
    for n in range(0, 40):
        for world in range(1, 8):
            sizes = split_sizes(n, world)
            assert sum(sizes) == n and max(sizes) - min(sizes) <= 1


def test_layout_matches_brute_force_and_remaps_ranges():
    from powershard.image_layout import Layout
    rng = random.Random(3)
    for _ in range(300):
        world = rng.randint(2, 7)
        rank = rng.randrange(world)
        segments = [(rng.randint(0, 9), False), (rng.randint(world, 60), True), (rng.randint(0, 9), False)]
        rng.shuffle(segments)
        lay = Layout(segments, world, rank)
        positions = brute_local_positions(segments, world, rank)
        assert lay.local_positions() == positions and lay.local_len == len(positions)
        for _ in range(10):
            s = rng.randint(0, lay.total)
            e = rng.randint(s, lay.total)
            # локальный интервал = локальные строки, попавшие в глобальный [s, e)
            inside = [i for i, p in enumerate(positions) if s <= p < e]
            got = lay.remap_ranges([(s, e, 7)])
            if inside:
                assert got == [(inside[0], inside[-1] + 1, 7)]
            else:
                assert got == []
            assert lay.local_index(s) == sum(1 for p in positions if p < s)


def test_layout_requires_one_split_segment():
    from powershard.image_layout import Layout
    with pytest.raises(ValueError):
        Layout([(3, False), (5, False)], 2, 0)
    with pytest.raises(ValueError):
        Layout([(3, True), (5, True)], 2, 0)


def test_choose_mode():
    from powershard.image_layout import choose_mode
    assert choose_mode("auto", True, 1, 6, 4096)[0] == "sequence"
    assert choose_mode("auto", False, 2, 6, 4096)[0] == "batch"
    assert choose_mode("auto", False, 1, 6, 4096)[0] == "replicate"
    assert choose_mode("auto", True, 2, 6, 100)[0] == "batch"            # мало токенов на 6 GPU
    assert choose_mode("sequence", False, 1, 6, 4096)[0] == "replicate"
    assert choose_mode("batch", True, 1, 6, 4096)[0] == "replicate"
    assert choose_mode("replicate", True, 4, 6, 4096)[0] == "replicate"
    assert choose_mode("auto", True, 1, 1, 4096)[0] == "replicate"


def _fake_folder_paths(monkeypatch):
    fake = types.ModuleType("folder_paths")
    fake.get_filename_list = lambda folder: [f"{folder}_a.safetensors"]
    fake.get_full_path_or_raise = lambda folder, name: f"/models/{folder}/{name}"
    fake.get_folder_paths = lambda folder: []
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    return fake


def test_image_nodes_register_without_torch():
    code = ("import sys; from powershard.common_nodes import COMMON_NODE_CLASS_MAPPINGS as n; "
            "assert {'PowerShardImageLoader', 'PowerShardCheckpointCLIPVAE', 'PowerShardLoRA'} <= set(n); "
            "import powershard.image_layout; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_image_loader_inputs(monkeypatch):
    _fake_folder_paths(monkeypatch)
    from powershard.image_nodes import PowerShardImageLoader
    fields = PowerShardImageLoader.INPUT_TYPES()
    assert "checkpoints/checkpoints_a.safetensors" in fields["required"]["checkpoint"][0]
    assert "diffusion_models/diffusion_models_a.safetensors" in fields["required"]["checkpoint"][0]
    assert fields["required"]["parallel"][0] == ["auto", "sequence", "batch", "replicate"]
    assert fields["optional"]["lora"][0] == "POWERSHARD_LORA"
    assert fields["optional"]["weights"][0] == "POWERSHARD_WEIGHTS"


def test_lora_type_is_shared_by_every_loader(monkeypatch):
    _fake_folder_paths(monkeypatch)
    from powershard.common_nodes import PowerShardLoRA
    from powershard.ltx_nodes import LTX_NODE_CLASS_MAPPINGS
    from powershard.wan_nodes import WAN_NODE_CLASS_MAPPINGS
    for cls in (WAN_NODE_CLASS_MAPPINGS["PowerShardWanLoader"], WAN_NODE_CLASS_MAPPINGS["PowerShardWan22MoELoader"],
                LTX_NODE_CLASS_MAPPINGS["PowerShardLTXLoader"], LTX_NODE_CLASS_MAPPINGS["PowerShardH3QuantLoader"]):
        assert cls.INPUT_TYPES()["optional"]["lora"][0] == "POWERSHARD_LORA"
    spec = PowerShardLoRA().add("loras_a.safetensors", 0.5)[0]
    assert len(spec) == 1 and spec[0].strength == 0.5 and spec[0].apply_to == "all"
    # старые Wan/LTX LoRA ноды скрыты, но открывают старые workflows
    assert WAN_NODE_CLASS_MAPPINGS["PowerShardWanLoRA"].DEPRECATED
    assert WAN_NODE_CLASS_MAPPINGS["PowerShardWanLoRA"].RETURN_TYPES == ("POWERSHARD_LORA",)
    assert LTX_NODE_CLASS_MAPPINGS["PowerShardLTXLoRA"].DEPRECATED


def test_text_encoder_image_types(monkeypatch):
    _fake_folder_paths(monkeypatch)
    from powershard.common_nodes import PowerShardTextEncoder, TEXT_ENCODERS
    types_ = PowerShardTextEncoder.INPUT_TYPES()["required"]["type"][0]
    for name in ("wan_umt5", "ltx_gemma", "flux", "flux2", "qwen_image", "z_image", "moss_image", "sdxl"):
        assert name in types_
    assert TEXT_ENCODERS["z_image"] == "lumina2" and TEXT_ENCODERS["sdxl"] == "stable_diffusion"
    optional = PowerShardTextEncoder.INPUT_TYPES()["optional"]
    assert optional["text_encoder_2"][0][0] == "none"
    paths = PowerShardTextEncoder._paths("text_encoders_a.safetensors", "flux", "none",
                                         ("text_encoders_a.safetensors", "none", "none"))
    assert len(paths) == 2
    with pytest.raises(ValueError, match="wan_umt5"):
        PowerShardTextEncoder().load("text_encoders_a.safetensors", "wan_umt5", "workers", config=object(),
                                     text_encoder_2="text_encoders_a.safetensors")


def test_weights_fp32_only_for_image():
    from powershard.common_nodes import PowerShardWeights, apply_weights
    from powershard.wan_config import WanOptions
    w = PowerShardWeights().create("dequantize", "auto", "fp32", False)[0]
    assert w.weight_dtype == "fp32"
    with pytest.raises(ValueError, match="image"):
        apply_weights(WanOptions(), w)


def test_decide_dtypes(request):
    torch = pytest.importorskip("torch")
    comfy_path = request.config.getoption("--comfy")
    if comfy_path is None:
        pytest.skip("NOT_RUN: pass --comfy for the image adapter dtype test")
    sys.path.insert(0, str(Path(comfy_path).resolve()))
    saved = sys.argv
    sys.argv = ["powershard-test", "--cpu", "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        from powershard.image_adapter import decide_dtypes
    finally:
        sys.argv = saved
    v100, rtx = [(7, 0)] * 6, [(12, 0)] * 2
    flux = [torch.bfloat16, torch.float16, torch.float32]
    zimage = [torch.bfloat16, torch.float32]
    assert decide_dtypes(flux, "auto", v100) == ("fp16", "fp16")
    assert decide_dtypes(flux, "auto", rtx) == ("bf16", "bf16")
    assert decide_dtypes(zimage, "auto", v100) == ("bf16", "fp32")      # bf16-модель на V100: точное FP32
    assert decide_dtypes(zimage, "fp16", v100) == ("fp16", "fp16")      # явный выбор пользователя
    assert decide_dtypes(zimage, "bf16", v100) == ("bf16", "fp32")
    assert decide_dtypes(zimage, "auto", v100[:1] + rtx) == ("bf16", "fp32")


def test_h3_pdd_shape(tmp_path):
    pytest.importorskip("torch")
    import json
    from wan_fixtures import write_safetensors
    from powershard.h3_quant import H3Checkpoint
    tensors = {"video_patch_proj.weight": ("F16", [32, 8], None), "adaln_t_table": ("F32", [17, 4], None),
               "blocks.0.attn.q_norm.weight": ("F16", [8], None), "blocks.0.attn.qkv_proj.weight": ("F16", [96, 32], None),
               "blocks.0.mlp.fc1.weight": ("F16", [128, 32], None), "token_refiner.blocks.0.x.weight": ("F16", [4], None),
               "condition_proj.weight": ("F16", [32, 24], None), "rope.inv_freq": ("F32", [1], None),
               "final_layer.video_out.weight": ("F32", [4 * 24 * 3, 32], None),
               "final_layer.audio_out.weight": ("F32", [32 * 3, 32], None)}
    path = write_safetensors(tmp_path / "h3.safetensors", tensors,
                             metadata={"config": json.dumps({"transformer": {"pdd_steps": 3}})})
    conf = H3Checkpoint(path).model_config()
    assert conf["pdd_steps"] == 3 and conf["latents_dim"] == 24 and conf["audio_latents_dim"] == 32
