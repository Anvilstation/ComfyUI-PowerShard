#!/usr/bin/env python3
"""Сквозная проверка PowerShard Wan на настоящих GPU: WanSession -> wan_worker -> FSDP2/NCCL.

Малая случайная Wan (по умолчанию dim=1280 -> 10 heads: Ulysses с паддингом heads на 3/4/6 GPU),
сравнение с native comfy WanModel FP32 на CPU. Это проверка execution/sequence/FSDP/fp8/LoRA/MoE,
НЕ качества настоящего Wan 2.2 checkpoint.

Примеры (из папки custom node, Python вашей ComfyUI):
  python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus all --sequence-mode ulysses --moe --fp8 --lora
  python scripts/probe_wan_cuda.py --comfy /opt/ComfyUI --gpus 0,1,2 --weight-placement cpu --moe --residency swap --ram-roundtrip
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
p.add_argument("--moe", action="store_true", help="два эксперта high/low в одной session")
p.add_argument("--residency", choices=["both", "swap"], default="both")
p.add_argument("--fp8", action="store_true", help="low-эксперт хранится как fp8_scaled")
p.add_argument("--lora", action="store_true", help="LoRA (up/down+alpha+diff_b) на high")
p.add_argument("--ram-roundtrip", action="store_true")
p.add_argument("--dim", type=int, default=1280)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--latent", default="5x30x52", help="кадры x H x W латента (480p ~ 21x60x104)")
p.add_argument("--variant", default="base",
               choices=["base", "i2v", "vace", "s2v", "animate", "camera", "uni3c", "humo", "scail", "scail2", "wandancer",
                        "wandancer30", "animate2", "multitalk1", "multitalk2"],
               help="base: T2V/MoE путь; остальные — малая модель варианта (VACE/S2V/Animate/Fun Camera/Uni3C, HuMo, "
                    "SCAIL/SCAIL-2, WanDancer fps 24/30, Animate2, InfiniteTalk 1/2 speakers)")
p.add_argument("--animate2-cache", action="store_true", help="Animate2: два шага с WanAnimate2Cache (pose inputs в RAM workers)")
p.add_argument("--output", default="reports/local-wan-cuda-probe.json")
a = p.parse_args()

from wan_reference import import_comfy, tiny_config, native_model, save_native  # noqa: E402
wan = import_comfy(a.comfy)
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from powershard.config import DistributedConfig  # noqa: E402
from powershard.wan_adapter import load_wan, EXPERT_KEY, UNI3C_KEY  # noqa: E402
from powershard.wan_config import WanOptions, WanLoraSpec  # noqa: E402

report = dict(status="FAIL", args=vars(a), real_checkpoint="NOT_RUN", checks=[])
folder = Path(tempfile.mkdtemp(prefix="powershard-wan-probe-"))
session = None
try:
    if a.variant != "base":
        import types
        from wan_reference import variant_call, our_options, uni3c_pair, save_uni3c_comfy, multitalk_pair
        from powershard.wan_adapter import MULTITALK_KEY
        config, build = variant_call(a.variant)
        native = native_model(wan, config, seed=5)
        path = save_native(native, folder / "variant.safetensors")
        dconfig = DistributedConfig(gpu_ids=a.gpus, weight_placement=a.weight_placement, sequence_mode=a.sequence_mode,
                                    sequence_comm_dtype=a.comm, attention_backend=a.attention_backend, reserve_gib=0,
                                    release_after_sampling=False)
        model_type = "animate2" if config["model_type"] == "animate2" else "auto"
        base = load_wan({"main": str(path)}, dconfig, WanOptions(fp16_safe=a.fp16_safe, model_type=model_type), (),
                        report_dir=folder / "reports")
        session = base.session
        args, kwargs, native_kwargs, native_options = build(native)
        options = dict(our_options(native_options), **{EXPERT_KEY: "main"})
        call_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")
                       and k not in ("uni3c_render", "multitalk_audio", "multitalk_masks")}
        if a.variant == "uni3c":  # те же веса (детерминированный seed) через файл model_patches
            native_cnet, _ = uni3c_pair(config["dim"])
            cnet_path = save_uni3c_comfy(native_cnet, folder / "uni3c.safetensors")
            st = cnet_path.stat()
            options[UNI3C_KEY] = dict(path=str(cnet_path), size=st.st_size, mtime_ns=st.st_mtime_ns, strength=.8,
                                      sigma_start=10., sigma_end=0., render=kwargs["uni3c_render"])
        if a.variant.startswith("multitalk"):
            native_blocks, _ = multitalk_pair(config["dim"], config["num_layers"])
            state = {"blocks." + k: v.half() for k, v in native_blocks.blocks.state_dict().items()}
            state["audio_proj.proj1.weight"] = torch.zeros((1, 1), dtype=torch.float16)  # метка формата patch
            patch_path = folder / "infinitetalk.safetensors"
            save_file(state, str(patch_path))
            st = patch_path.stat()
            options[MULTITALK_KEY] = dict(path=str(patch_path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                                          audio_scale=kwargs["_powershard_multitalk"]["audio_scale"],
                                          audio=kwargs["multitalk_audio"], masks=kwargs.get("multitalk_masks"))
        steps = 2 if a.variant == "animate2" and a.animate2_cache else 1
        if steps == 2:
            options["animate2_cache"] = types.SimpleNamespace(dtype="default")
        with torch.no_grad():
            expected = native(*args, transformer_options=dict(native_options), **native_kwargs)
            for step in range(steps):
                started = time.perf_counter()
                actual = base.model.diffusion_model(*args, transformer_options=dict(options), **call_kwargs)
                wall = time.perf_counter() - started
                error, scale = float((actual.cpu() - expected).abs().max()), float(expected.abs().max())
                metrics = session.history[-1]["ranks"][0]["metrics"]
                report["checks"].append(dict(expert=f"{a.variant}#{step}", max_abs=error, reference_max=scale,
                                             ok=error <= .04 + .04 * scale, wall_s=wall, forward_s=metrics.get("forward_s"),
                                             attention=metrics.get("attention", {}).get("effective_used"),
                                             sequence=metrics.get("sequence_communication"), uni3c=metrics.get("uni3c"),
                                             multitalk=metrics.get("multitalk")))
        report["status"] = "PASS" if all(c["ok"] for c in report["checks"]) else "FAIL_NUMERIC"
        raise SystemExit(0)
    config_kwargs = tiny_config(dim=a.dim, num_heads=a.dim // 128, ffn_dim=a.dim * 2, num_layers=a.layers, text_dim=64)
    natives = {"high": native_model(wan, config_kwargs, seed=1)}
    if a.moe:
        natives["low"] = native_model(wan, config_kwargs, seed=2)
    paths = {}
    for slot, native in natives.items():
        path = folder / f"{slot}.safetensors"
        if slot == "low" and a.fp8:
            state, fp8 = native.state_dict(), {}
            for name, value in state.items():
                if name.startswith("blocks.") and name.endswith(".weight") and value.ndim == 2:
                    scale = value.abs().amax().clamp(min=1e-12) / 448.
                    fp8[name] = (value / scale).to(torch.float8_e4m3fn)
                    fp8[name[:-len(".weight")] + ".scale_weight"] = scale.float().reshape(())
                    # Reference использует ровно деквантованные веса.
                    with torch.no_grad():
                        dict(native.named_parameters())[name].copy_(fp8[name].float() * scale)
                else:
                    fp8[name] = value.half()
            fp8["scaled_fp8"] = torch.zeros(0, dtype=torch.float8_e4m3fn)
            save_file(fp8, str(path))
        else:
            save_native(native, path)
        paths[slot] = str(path)
    loras = ()
    if a.lora:
        g = torch.Generator().manual_seed(3)
        up, down = torch.randn((a.dim, 8), generator=g) * .05, torch.randn((8, a.dim), generator=g) * .05
        diff_b = torch.randn((a.dim,), generator=g) * .05
        lora_path = folder / "lora.safetensors"
        save_file({"diffusion_model.blocks.0.self_attn.q.lora_up.weight": up.half(),
                   "diffusion_model.blocks.0.self_attn.q.lora_down.weight": down.half(),
                   "diffusion_model.blocks.0.self_attn.q.alpha": torch.tensor(4.),
                   "diffusion_model.blocks.0.self_attn.q.diff_b": diff_b.half()}, str(lora_path))
        with torch.no_grad():
            params = dict(natives["high"].named_parameters())
            params["blocks.0.self_attn.q.weight"].add_(.8 * (up.half().float() @ down.half().float()) * (4. / 8))
            params["blocks.0.self_attn.q.bias"].add_(.8 * diff_b.half().float())
        loras = (WanLoraSpec(str(lora_path), .8, "high_noise" if a.moe else "all"),)
    dconfig = DistributedConfig(gpu_ids=a.gpus, weight_placement=a.weight_placement, sequence_mode=a.sequence_mode,
                                sequence_comm_dtype=a.comm, attention_backend=a.attention_backend, reserve_gib=0,
                                release_after_sampling=False)
    options = WanOptions(fp16_safe=a.fp16_safe, mlp_chunk_mode="auto", moe_residency=a.residency)
    experts = paths if a.moe else {"main": paths["high"]}
    start = time.perf_counter()
    base = load_wan(experts, dconfig, options, loras, report_dir=folder / "reports")
    session = base.session
    proxy = base.model.diffusion_model
    frames, h, w = (int(v) for v in a.latent.split("x"))
    g = torch.Generator().manual_seed(7)
    x = torch.randn((2, 16, frames, h, w), generator=g)
    ctx = torch.randn((2, 12, 64), generator=g)
    t = torch.tensor([950., 950.])
    order = ["high", "low", "high"] if a.moe else ["main"]
    for index, slot in enumerate(order):
        if a.ram_roundtrip and index == len(order) - 1:
            session.idle()
            report["ram_roundtrip"] = dict(idle_on_cpu=session.idle_on_cpu, last_memory=session.last_memory)
        with torch.no_grad():
            started = time.perf_counter()
            actual = proxy(x, t, ctx, transformer_options={EXPERT_KEY: slot})
            wall = time.perf_counter() - started
            expected = natives["high" if slot == "main" else slot](x, t, ctx, transformer_options={})
        error = float((actual.cpu() - expected).abs().max())
        scale = float(expected.abs().max())
        ranks = session.history[-1]["ranks"]
        metrics = ranks[0]["metrics"]
        report["checks"].append(dict(expert=slot, max_abs=error, reference_max=scale, ok=error <= .03 + .03 * scale,
                                     wall_s=wall, forward_s=metrics.get("forward_s"),
                                     attention=metrics.get("attention", {}).get("effective_used"),
                                     sequence=metrics.get("sequence_communication"),
                                     expert_switch=metrics.get("expert_switch"), phase_cache=metrics.get("phase_cache"),
                                     memory=[r["metrics"]["memory"] for r in ranks],
                                     weight_memory=metrics.get("weight_memory")))
    report["load_and_checks_s"] = time.perf_counter() - start
    report["preflight"] = getattr(session, "preflight", None)
    report["status"] = "PASS" if all(c["ok"] for c in report["checks"]) else "FAIL_NUMERIC"
finally:
    if session is not None:
        session.close()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(dict(status=report["status"], checks=[{k: c[k] for k in ("expert", "max_abs", "ok", "forward_s")}
                                                           for c in report["checks"]], output=a.output),
                     ensure_ascii=False, indent=1))
