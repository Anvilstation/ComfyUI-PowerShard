#!/usr/bin/env python3
"""Сквозная проверка PowerShard LTX на GPU: WanSession(role=ltx) -> wan_worker -> FSDP2/NCCL.

Малая случайная LTXAV (FP32 native comfy на CPU как эталон, in_training=True — без comfy-kitchen) сохраняется
как полный checkpoint (model.diffusion_model.* + metadata config), грузится PowerShard и сравнивается:
обычный шаг (I2V-маска timestep), preprocess коннекторов, guides с маской внимания (strength<1),
STG/модальность, Block Cache (повтор шага). Это проверка исполнения/sequence/FSDP, НЕ качества LTX-2.

  python scripts/probe_ltx_cuda.py --comfy /opt/ComfyUI --gpus all --sequence-mode ulysses
  python scripts/probe_ltx_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2 --weight-placement cpu --sequence-mode token --fp8
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

p = argparse.ArgumentParser()
p.add_argument("--comfy", required=True)
p.add_argument("--gpus", default="all")
p.add_argument("--weight-placement", choices=["gpu", "cpu", "ats"], default="gpu")
p.add_argument("--sequence-mode", choices=["token", "ulysses"], default="ulysses")
p.add_argument("--comm", choices=["fp32", "fp16"], default="fp32")
p.add_argument("--attention-backend", default="auto", choices=["auto", "vllm_flash_attn", "flash_attn", "sdpa", "math"])
p.add_argument("--fp16-safe", action="store_true")
p.add_argument("--fp8", action="store_true", help="Linear-веса блоков как fp8_scaled")
p.add_argument("--adaln", action="store_true", help="LTX 2.3+: cross_attention_adaln + gated attention")
p.add_argument("--heads", type=int, default=6)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--latent", default="4x8x12", help="кадры x H x W видео-латента")
p.add_argument("--output", default="reports/local-ltx-cuda-probe.json")
a = p.parse_args()

from wan_reference import import_comfy, randomize  # noqa: E402
import_comfy(a.comfy)
import torch  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.lightricks.av_model import LTXAVModel  # noqa: E402
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from powershard.accel import BlockCacheHolder  # noqa: E402
from powershard.config import DistributedConfig  # noqa: E402
from powershard.ltx_adapter import load_ltx  # noqa: E402
from powershard.ltx_config import LTXOptions  # noqa: E402

comfy.model_management.in_training = True
report = dict(status="FAIL", args=vars(a), real_checkpoint="NOT_RUN", checks=[])
folder = Path(tempfile.mkdtemp(prefix="powershard-ltx-probe-"))
session = None
try:
    config = dict(in_channels=16, audio_in_channels=128, num_layers=a.layers, attention_head_dim=128,
                  num_attention_heads=a.heads, cross_attention_dim=128 * a.heads, audio_attention_head_dim=64,
                  audio_num_attention_heads=4, audio_cross_attention_dim=256, caption_channels=96,
                  connector_attention_head_dim=32, connector_num_attention_heads=3, connector_num_layers=1, rope_type="split")
    if a.adaln:
        config.update(cross_attention_adaln=True, apply_gated_attention=True, caption_proj_before_connector=True,
                      connector_attention_head_dim=128, connector_num_attention_heads=a.heads,
                      audio_connector_attention_head_dim=64, audio_connector_num_attention_heads=4)
    native = randomize(LTXAVModel(**config, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init), 11)
    with torch.no_grad():
        for name, param in native.named_parameters():
            if "scale_shift_table" in name or "learnable_registers" in name:
                param.copy_(.1 * torch.randn(param.shape, generator=torch.Generator().manual_seed(len(name))))
    state = {}
    for name, value in native.state_dict().items():
        key = "model.diffusion_model." + name
        if a.fp8 and name.startswith("transformer_blocks.") and name.endswith(".weight") and value.ndim == 2:
            scale = value.abs().amax().clamp(min=1e-12) / 448.
            q = (value / scale).to(torch.float8_e4m3fn)
            state[key] = q
            state[key[:-len(".weight")] + ".weight_scale"] = scale.float().reshape(())
            with torch.no_grad():
                dict(native.named_parameters())[name].copy_(q.float() * scale)
        else:
            state[key] = value.to(torch.bfloat16)
            with torch.no_grad():
                if name in dict(native.named_parameters()):
                    dict(native.named_parameters())[name].copy_(value.to(torch.bfloat16).float())
    state["vae.dummy"] = torch.zeros(1)                      # полный checkpoint: не-DiT ключи игнорируются
    path = folder / "ltx_tiny.safetensors"
    save_file(state, str(path), metadata={"config": json.dumps({"transformer": config})})
    dconfig = DistributedConfig(gpu_ids=a.gpus, weight_placement=a.weight_placement, sequence_mode=a.sequence_mode,
                                sequence_comm_dtype=a.comm, attention_backend=a.attention_backend, reserve_gib=0,
                                release_after_sampling=False)
    started = time.perf_counter()
    patcher = load_ltx(str(path), dconfig, LTXOptions(fp16_safe=a.fp16_safe, mlp_chunk_mode="auto"), (),
                       report_dir=folder / "reports")
    session = patcher.session
    proxy = patcher.model.diffusion_model
    frames, h, w = (int(v) for v in a.latent.split("x"))
    g = torch.Generator().manual_seed(7)
    vx = torch.randn((2, 16, frames, h, w), generator=g)
    ax = torch.randn((2, 8, 2 * frames + 3, 16), generator=g)
    tokens = frames * h * w
    vt = torch.full((2, tokens, 1), .8)
    vt[:, :h * w] = 0.
    at = torch.tensor([.8, .8])
    ctx_raw = torch.randn((2, 9, config["caption_channels"]), generator=g)

    def check(label, actual, expected, **extra):
        pairs = list(zip(actual, expected)) if isinstance(expected, (list, tuple)) else [(actual, expected)]
        error = max(float((x.cpu().float() - y.float()).abs().max()) for x, y in pairs)
        scale = max(float(y.abs().max()) for _, y in pairs)
        metrics = session.history[-1]["ranks"][0]["metrics"] if session.history and "ranks" in session.history[-1] else {}
        report["checks"].append(dict(label=label, max_abs=error, reference_max=scale, ok=error <= .04 + .04 * scale,
                                     forward_s=metrics.get("forward_s"), sequence=metrics.get("sequence_communication"),
                                     attention=metrics.get("attention", {}).get("effective_used"),
                                     block_cache=metrics.get("block_cache"), **extra))

    with torch.no_grad():
        ctx_native = native.preprocess_text_embeds(ctx_raw, unprocessed=True)
        ctx = proxy.preprocess_text_embeds(ctx_raw, unprocessed=True)
        check("preprocess_connectors", ctx, ctx_native)
        expected = native([vx.clone(), ax.clone()], (vt, at), ctx_native, frame_rate=24, transformer_options={})
        actual = proxy([vx, ax], (vt, at), ctx_native, frame_rate=24, transformer_options={})
        check("forward_i2v_mask", actual, expected)
        for label, ours, theirs in (("stg_block1", {"stg_self_attn_blocks": frozenset({1})}, None),
                                    ("modality_off", {"a2v_cross_attn": False, "v2a_cross_attn": False}, None)):
            expected = native([vx.clone(), ax.clone()], (vt, at), ctx_native, frame_rate=24, transformer_options=dict(ours))
            actual = proxy([vx, ax], (vt, at), ctx_native, frame_rate=24, transformer_options=dict(ours))
            check(label, actual, expected)
        # Guide (keyframe) кадр в конце + strength 0.5 -> GuideAttentionMask (token-путь с маской по глобальным строкам).
        guide = torch.randn((2, 16, 1, h, w), generator=g)
        vxg = torch.cat([vx, guide], dim=2)
        coords = native.patchifier.get_latent_coords(1, h, w, 2, torch.device("cpu"))
        keyframe_idxs = latent_to_pixel_coords(coords, native.vae_scale_factors, native.causal_temporal_positioning)
        denoise = torch.ones((2, 1, frames + 1, h, w))
        denoise[:, :, -1] = 0.
        vtg = torch.cat([vt, torch.zeros((2, h * w, 1))], dim=1)
        entries = [dict(pre_filter_count=h * w, strength=0.5, pixel_mask=None, latent_shape=[1, h, w])]
        kwargs = dict(keyframe_idxs=keyframe_idxs.float(), denoise_mask=denoise, guide_attention_entries=entries)
        expected = native([vxg.clone(), ax.clone()], (vtg, at), ctx_native, frame_rate=24, transformer_options={}, **kwargs)
        actual = proxy([vxg, ax], (vtg, at), ctx_native, frame_rate=24, transformer_options={}, **kwargs)
        check("guides_attention_mask", actual, expected)
        # Block Cache: тот же шаг дважды -> второй пропускает блоки 1..N и даёт тот же результат.
        holder = BlockCacheHolder(0.05, sigma_start=10., sigma_end=0., max_skips=3, warmup_steps=0)
        options = {"powershard_block_cache": holder, "cond_or_uncond": [0, 1]}
        expected = native([vx.clone(), ax.clone()], (vt, at), ctx_native, frame_rate=24, transformer_options={})
        for step in range(2):
            actual = proxy([vx, ax], (vt, at), ctx_native, frame_rate=24, transformer_options=dict(options))
            check(f"block_cache#{step}", actual, expected)
    report["load_and_checks_s"] = time.perf_counter() - started
    report["preflight"] = getattr(session, "preflight", None)
    cache_skipped = report["checks"][-1].get("block_cache") or {}
    report["block_cache_second_call_skipped"] = cache_skipped.get("skipped_this_call")
    report["status"] = "PASS" if all(c["ok"] for c in report["checks"]) else "FAIL_NUMERIC"
finally:
    if session is not None:
        session.close()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(dict(status=report["status"], checks=[{k: c[k] for k in ("label", "max_abs", "ok", "forward_s")}
                                                           for c in report["checks"]], output=a.output),
                     ensure_ascii=False, indent=1))
