"""Wan: заголовки, геометрия, fp8, LoRA planning, workflows. Без torch/CUDA."""
import json
import sys
from pathlib import Path
import pytest
from wan_fixtures import write_tiny_wan, write_safetensors, tiny_wan_shapes

ROOT = Path(__file__).resolve().parents[1]


def test_infer_t2v_geometry_and_prefixes(tmp_path):
    from powershard.wan_config import WanCheckpoint, describe_family, comfy_unet_config
    for prefix in ("", "diffusion_model.", "model.diffusion_model."):
        ckpt = WanCheckpoint(write_tiny_wan(tmp_path / f"t{len(prefix)}.safetensors", prefix=prefix))
        config = ckpt.model_config()
        assert ckpt.prefix == prefix
        assert config["dim"] == 256 and config["num_heads"] == 2 and config["num_layers"] == 2
        assert config["ffn_dim"] == 512 and config["in_dim"] == 16 and config["out_dim"] == 16
        assert config["text_dim"] == 32 and config["freq_dim"] == 16 and config["model_type"] == "t2v"
        assert config["cross_attn_norm"] and config["qk_norm"]
        assert describe_family(config) == "wan_t2v"
        unet = comfy_unet_config(config)
        assert unet["image_model"] == "wan2.1" and unet["disable_unet_model_creation"]
        assert "blocks.0.self_attn.q.weight" in ckpt.tensors
        assert ckpt.tensors["blocks.0.self_attn.q.weight"]["key"] == prefix + "blocks.0.self_attn.q.weight"


def test_families(tmp_path):
    from powershard.wan_config import WanCheckpoint, describe_family
    five = WanCheckpoint(write_tiny_wan(tmp_path / "5b.safetensors", in_dim=48, out_dim=48)).model_config()
    assert describe_family(five) == "wan2.2_ti2v_5b"
    i2v22 = WanCheckpoint(write_tiny_wan(tmp_path / "i2v22.safetensors", in_dim=36)).model_config()
    assert describe_family(i2v22) == "wan2.2_i2v_a14b_expert" and i2v22["model_type"] == "t2v"
    i2v21 = WanCheckpoint(write_tiny_wan(tmp_path / "i2v21.safetensors", in_dim=36, i2v=True)).model_config()
    assert i2v21["model_type"] == "i2v"
    ref = WanCheckpoint(write_tiny_wan(tmp_path / "ref.safetensors", ref=True)).model_config()
    assert ref["in_dim_ref_conv"] == 16


def test_fp8_scaled_storage_and_memory_plan(tmp_path):
    from powershard.wan_config import WanCheckpoint, memory_plan
    fp16 = WanCheckpoint(write_tiny_wan(tmp_path / "fp16.safetensors"))
    fp8 = WanCheckpoint(write_tiny_wan(tmp_path / "fp8.safetensors", fp8_scaled=True))
    assert fp16.storage()["kind"] == "fp16" and fp8.storage()["kind"] == "fp8_scaled"
    assert fp8.fp8_scale_key("blocks.0.self_attn.q.weight") == "blocks.0.self_attn.q.scale_weight"
    assert fp8.is_metadata("scaled_fp8") and not fp8.quantization()
    # fp8 веса деквантуются в FP16 -> постоянные байты как у fp16 (scales не считаются).
    assert memory_plan(fp8, 6)["converted_storage_bytes"] == memory_plan(fp16, 6)["converted_storage_bytes"]
    plan = memory_plan(fp16, 6)
    assert plan["largest_group_bytes_upper_bound"] > 0
    assert plan["shard_bytes_lower_bound"] * 6 >= plan["converted_storage_bytes"]


def test_new_variants_detection(tmp_path):
    from powershard.wan_config import WanCheckpoint, describe_family, comfy_unet_config, WanOptions
    humo = {"audio_proj.audio_proj_glob_1.layer.bias": ("F16", [512], None),
            "audio_proj.audio_proj_glob_3.layer.weight": ("F16", [16 * 1536, 512], None),
            "audio_proj.audio_proj_glob_norm.layer.weight": ("F16", [1536], None)}
    config = WanCheckpoint(write_tiny_wan(tmp_path / "humo.safetensors", extra=humo)).model_config()
    assert config["model_type"] == "humo" and describe_family(config) == "wan_humo"
    bad = dict(humo, **{"audio_proj.audio_proj_glob_3.layer.weight": ("F16", [8 * 1536, 512], None)})
    with pytest.raises(ValueError, match="16"):
        WanCheckpoint(write_tiny_wan(tmp_path / "humo8.safetensors", extra=bad)).model_config()
    scail = {"patch_embedding_pose.weight": ("F32", [256, 20, 1, 2, 2], None)}
    config = WanCheckpoint(write_tiny_wan(tmp_path / "scail.safetensors", in_dim=20, i2v=True, extra=scail)).model_config()
    assert config["model_type"] == "scail" and comfy_unet_config(config)["model_type"] == "scail"
    scail2 = dict(scail, **{"patch_embedding_mask.weight": ("F32", [256, 28, 1, 2, 2], None)})
    config = WanCheckpoint(write_tiny_wan(tmp_path / "scail2.safetensors", in_dim=20, i2v=True, extra=scail2)).model_config()
    assert config["model_type"] == "scail2" and config["mask_in_dim"] == 28
    dancer = {"patch_embedding_global.weight": ("F32", [256, 36, 1, 2, 2], None),
              "music_projection.weight": ("F16", [256, 35], None), "music_encoder.0.norm1.weight": ("F16", [256], None),
              "music_encoder.0.self_attn.in_proj_weight": ("F16", [768, 256], None),
              "music_encoder.0.self_attn.in_proj_bias": ("F16", [768], None)}
    dancer.update({f"music_injector.injector.{i}.q.weight": ("F16", [256, 256], None) for i in range(8)})
    ckpt = WanCheckpoint(write_tiny_wan(tmp_path / "dancer.safetensors", in_dim=36, i2v=True, extra=dancer))
    config = ckpt.model_config()
    assert config["model_type"] == "wandancer" and config["music_feature_dim"] == 35
    # fused in_proj -> q/k/v_proj со смещением строк (как comfy process_unet_state_dict)
    k = ckpt.tensors["music_encoder.0.self_attn.k_proj.weight"]
    assert k["shape"] == [256, 256] and k["row_offset"] == 256 and k["key"].endswith("in_proj_weight")
    assert ckpt.tensors["music_encoder.0.self_attn.v_proj.bias"]["row_offset"] == 512
    assert "music_encoder.0.self_attn.in_proj_weight" not in ckpt.tensors
    # Animate2: форма Wan2.1 I2V; распознаётся по metadata или опцией loader.
    i2v = write_tiny_wan(tmp_path / "i2v.safetensors", in_dim=36, i2v=True)
    assert WanCheckpoint(i2v).model_config()["model_type"] == "i2v"
    assert WanCheckpoint(i2v, "animate2").model_config()["model_type"] == "animate2"
    shapes = {k: ("F16", v, None) for k, v in tiny_wan_shapes(in_dim=36, i2v=True).items()}
    meta = write_safetensors(tmp_path / "a2.safetensors", shapes,
                             metadata={"config": json.dumps({"transformer": {"model_type": "animate2"}})})
    config = WanCheckpoint(meta).model_config()
    assert config["model_type"] == "animate2" and describe_family(config) == "wan_animate2"
    with pytest.raises(ValueError, match="Animate2"):
        WanCheckpoint(write_tiny_wan(tmp_path / "t2v.safetensors"), "animate2").model_config()
    assert WanOptions(model_type="animate2").model_type == "animate2"
    with pytest.raises(ValueError):
        WanOptions(model_type="humo")
    with pytest.raises(ValueError, match="head.modulation"):
        WanCheckpoint(write_safetensors(tmp_path / "x.safetensors", {"w": ("F16", [2], None)}))


def test_multitalk_patch_checkpoint(tmp_path):
    from powershard.wan_config import MultiTalkCheckpoint
    tensors = {"audio_proj.proj1.weight": ("F16", [512, 5 * 12 * 768], None), "audio_proj.norm.weight": ("F16", [768], None)}
    for i in range(3):
        tensors[f"blocks.{i}.audio_cross_attn.proj.weight"] = ("F16", [5120, 5120], None)
        tensors[f"blocks.{i}.audio_cross_attn.kv_linear.weight"] = ("F16", [10240, 768], None)
        tensors[f"blocks.{i}.norm_x.weight"] = ("F16", [5120], None)
    ckpt = MultiTalkCheckpoint(write_safetensors(tmp_path / "it.safetensors", tensors))
    assert ckpt.model_config() == dict(in_dim=5120, out_dim=768, num_layers=3)
    assert not any(name.startswith("audio_proj.") for name in ckpt.tensors)  # audio_proj остаётся на host
    with pytest.raises(ValueError, match="InfiniteTalk"):
        MultiTalkCheckpoint(write_tiny_wan(tmp_path / "w.safetensors"))


def test_variant_detection(tmp_path):
    from powershard.wan_config import WanCheckpoint, describe_family, comfy_unet_config
    vace = {"vace_patch_embedding.weight": ("F16", [256, 96, 1, 2, 2], None)}
    for i in range(2):
        vace[f"vace_blocks.{i}.after_proj.weight"] = ("F16", [256, 256], None)
    config = WanCheckpoint(write_tiny_wan(tmp_path / "vace.safetensors", extra=vace)).model_config()
    assert config["model_type"] == "vace" and config["vace_layers"] == 2 and config["vace_in_dim"] == 96
    assert describe_family(config) == "wan_vace" and comfy_unet_config(config)["model_type"] == "vace"
    camera = {"control_adapter.conv.weight": ("F16", [256, 24 * 64, 2, 2], None)}
    config = WanCheckpoint(write_tiny_wan(tmp_path / "cam.safetensors", in_dim=36, extra=camera)).model_config()
    assert config["model_type"] == "camera_2.2" and config["in_dim_control_adapter"] == 24
    s2v = {"casual_audio_encoder.encoder.final_linear.weight": ("F16", [256, 256], None)}
    s2v.update({f"audio_injector.injector.{i}.q.weight": ("F16", [256, 256], None) for i in range(12)})
    config = WanCheckpoint(write_tiny_wan(tmp_path / "s2v.safetensors", extra=s2v)).model_config()
    assert config["model_type"] == "s2v" and describe_family(config) == "wan2.2_s2v"
    animate = {"face_adapter.fuser_blocks.0.k_norm.weight": ("F16", [128], None)}
    config = WanCheckpoint(write_tiny_wan(tmp_path / "anim.safetensors", in_dim=36, extra=animate)).model_config()
    assert config["model_type"] == "animate"


def test_uni3c_checkpoint_renames_and_geometry(tmp_path):
    from powershard.wan_config import Uni3CCheckpoint
    tensors = {"controlnet_patch_embedding.weight": ("F16", [5120, 36, 1, 2, 2], None),
               "controlnet_mask_embedding.mask_proj.0.weight": ("F16", [256, 7, 4, 8, 8], None),
               "proj_in.weight": ("F16", [1024, 5120], None),
               "controlnet_blocks.0.ffn.0.bias": ("F16", [8192], None),
               "controlnet_blocks.0.norm1.linear.weight": ("F16", [3072, 5120], None),
               "controlnet_blocks.0.self_attn.to_q.weight": ("F16", [1024, 1024], None),
               "controlnet_blocks.0.self_attn.to_out.0.weight": ("F16", [1024, 1024], None),
               "proj_out.0.weight": ("F16", [5120, 1024], None), "proj_out.1.weight": ("F16", [5120, 1024], None)}
    ckpt = Uni3CCheckpoint(write_safetensors(tmp_path / "u.safetensors", tensors))
    assert "controlnet_blocks.0.self_attn.q.weight" in ckpt.tensors
    assert ckpt.tensors["controlnet_blocks.0.self_attn.o.weight"]["key"] == "controlnet_blocks.0.self_attn.to_out.0.weight"
    geometry = ckpt.model_config()
    assert geometry == dict(in_channels=36, conv_out_dim=5120, dim=1024, ffn_dim=8192, num_layers=2, time_embed_dim=5120,
                            out_proj_dim=5120, add_channels=7, mid_channels=256)


def test_options_and_lora_spec_validation(tmp_path):
    from powershard.wan_config import WanOptions, WanLoraSpec, lora_plan_for
    with pytest.raises(ValueError):
        WanOptions(mlp_chunk_mode="bad")
    with pytest.raises(ValueError):
        WanOptions(moe_residency="bad")
    lora = write_safetensors(tmp_path / "l.safetensors", {"diffusion_model.blocks.0.self_attn.q.lora_down.weight": ("F16", [4, 256], None),
                                                          "diffusion_model.blocks.0.self_attn.q.lora_up.weight": ("F16", [256, 4], None)})
    high = WanLoraSpec(str(lora), 1.0, "high_noise")
    both = WanLoraSpec(str(lora), .5, "all")
    zero = WanLoraSpec(str(lora), 0., "all")
    assert [x["apply_to"] for x in lora_plan_for("high", (high, both, zero))] == ["high_noise", "all"]
    assert [x["apply_to"] for x in lora_plan_for("low", (high, both))] == ["all"]
    with pytest.raises(ValueError, match="MoE"):
        lora_plan_for("main", (high,))


def lora_header(entries):
    return {k: {"dtype": "F16", "shape": s, "data_offsets": [0, 0]} for k, s in entries.items()}


def test_lora_planning_formats():
    from powershard.wan_lora import plan_lora
    model = {k: {"shape": s} for k, s in tiny_wan_shapes().items()}
    plan = plan_lora(lora_header({
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": [8, 256],
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": [256, 8],
        "diffusion_model.blocks.0.self_attn.q.alpha": [],
        "diffusion_model.blocks.0.self_attn.q.diff_b": [256],
        "diffusion_model.blocks.0.cross_attn.norm_k.diff": [256],
        "diffusion_model.blocks.1.modulation.diff": [1, 6, 256],
        "lora_unet_blocks_1_ffn_0.lora_down.weight": [8, 256],
        "lora_unet_blocks_1_ffn_0.lora_up.weight": [512, 8],
        "diffusion_model.blocks.1.ffn.2.lora_A.weight": [8, 512],
        "diffusion_model.blocks.1.ffn.2.lora_B.weight": [256, 8],
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": [8, 768]}), model)
    entries = plan["entries"]
    assert entries["blocks.0.self_attn.q.weight"]["alpha"].endswith(".alpha")
    assert entries["blocks.0.self_attn.q.bias"]["diff"].endswith(".diff_b")
    assert entries["blocks.0.cross_attn.norm_k.weight"]["diff"].endswith(".diff")
    assert entries["blocks.1.modulation"]["diff"].endswith("modulation.diff")
    assert entries["blocks.1.ffn.0.weight"]["up"].startswith("lora_unet_")
    assert entries["blocks.1.ffn.2.weight"]["down"].endswith("lora_A.weight")
    assert plan["ignored"] and plan["ranks"] == [8]


def test_lora_planning_rejects_mismatch():
    from powershard.wan_lora import plan_lora
    model = {k: {"shape": s} for k, s in tiny_wan_shapes().items()}
    with pytest.raises(ValueError, match="не соответствуют"):
        plan_lora(lora_header({"diffusion_model.blocks.9.self_attn.q.lora_down.weight": [4, 256],
                               "diffusion_model.blocks.9.self_attn.q.lora_up.weight": [256, 4]}), model)
    with pytest.raises(ValueError, match="форма"):
        plan_lora(lora_header({"diffusion_model.blocks.0.self_attn.q.lora_down.weight": [4, 512],
                               "diffusion_model.blocks.0.self_attn.q.lora_up.weight": [256, 4]}), model)
    with pytest.raises(ValueError, match="LoHa"):
        plan_lora(lora_header({"diffusion_model.blocks.0.self_attn.q.hada_w1_a": [4, 256]}), model)
    with pytest.raises(ValueError, match="up/down"):
        plan_lora(lora_header({"diffusion_model.blocks.0.self_attn.q.lora_up.weight": [256, 4]}), model)


@pytest.fixture
def comfy_folder_paths(request):
    path = request.config.getoption("--comfy")
    if path is None:
        pytest.skip("NOT_RUN: передайте --comfy для проверки INPUT_TYPES Wan nodes")
    sys.path.insert(0, str(Path(path).resolve()))
    import folder_paths
    return folder_paths


def test_wan_nodes_register_without_torch():
    import subprocess
    code = ("import sys; from powershard.wan_nodes import WAN_NODE_CLASS_MAPPINGS as n; "
            "assert {'PowerShardWan22MoELoader','PowerShardWanLoader','PowerShardWanLoRA','PowerShardWanOptions','PowerShardWanUni3C','PowerShardWanInfiniteTalk','PowerShardWanT5Distributed','PowerShardFreeVRAM','PowerShardWanAnimate2Cache'} <= n.keys(); "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_shipped_wan_workflows(comfy_folder_paths):
    from powershard.nodes import NODE_CLASS_MAPPINGS
    from powershard.wan_nodes import WAN_NODE_CLASS_MAPPINGS
    from powershard.workflow_migration import migrate_api, migrate_ui
    mappings = {**NODE_CLASS_MAPPINGS, **WAN_NODE_CLASS_MAPPINGS}
    files = sorted((ROOT / "workflows_wan").glob("*.api.json"))
    assert files
    for path in files:
        graph = json.loads(path.read_text())
        assert migrate_api(graph) == (graph, [])
        for key, node in graph.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    assert value[0] in graph, (path.name, key)
            cls = mappings.get(node["class_type"])
            if cls is None:
                continue
            fields = cls.INPUT_TYPES()
            required = set(fields.get("required", {}))
            assert required <= set(node["inputs"]) <= required | set(fields.get("optional", {})), (path.name, key)
    for path in sorted((ROOT / "workflows_wan").glob("*.ui.json")):
        graph = json.loads(path.read_text())
        assert migrate_ui(graph) == (graph, [])


def test_umt5_checkpoint_prefixes(tmp_path):
    """Distributed umT5: имена файла (без префикса / comfy-префикс) -> umt5xxl.transformer.*, spiece_model — tokenizer."""
    from powershard.wan_config import T5Checkpoint
    for prefix in ("", "umt5xxl.transformer."):
        tensors = {prefix + "shared.weight": ("F16", [256384, 64], None),
                   prefix + "encoder.final_layer_norm.weight": ("F16", [64], None),
                   "spiece_model": ("U8", [16], None)}
        for i in range(3):
            tensors[prefix + f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = ("F16", [64, 64], None)
        ckpt = T5Checkpoint(write_safetensors(tmp_path / f"t5{len(prefix)}.safetensors", tensors))
        assert ckpt.model_config() == dict(vocab_size=256384, d_model=64, num_layers=3)
        assert "umt5xxl.transformer.encoder.block.2.layer.0.SelfAttention.q.weight" in ckpt.tensors
        assert "spiece_model" in ckpt.ignored
