"""Sequence / batch parallel image-моделей против обычного forward (CPU, gloo, 2-3 процесса, родные модели ComfyUI).

  python -m pytest tests/test_image_parallel_torch.py --comfy /path/to/ComfyUI
"""
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _comfy(path):
    sys.path.insert(0, str(Path(path).resolve()))
    sys.path.insert(0, str(ROOT))
    saved = sys.argv
    sys.argv = ["powershard-test", "--cpu", "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.ops  # noqa: F401
    finally:
        sys.argv = saved


def _build(family):
    import comfy.ops
    ops = comfy.ops.disable_weight_init
    if family == "flux":
        from comfy.ldm.flux.model import Flux
        model = Flux(image_model="flux", in_channels=4, out_channels=4, vec_in_dim=8, context_in_dim=16, hidden_size=64,
                     mlp_ratio=2.0, num_heads=4, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000,
                     patch_size=2, qkv_bias=True, guidance_embed=False, txt_ids_dims=[], dtype=torch.float32,
                     device="cpu", operations=ops)
        x = torch.randn(3, 4, 16, 20)
        kwargs = dict(context=torch.randn(3, 7, 16), y=torch.randn(3, 8))
    else:
        from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
        model = QwenImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=4, num_layers=2,
                                            attention_head_dim=16, num_attention_heads=4, joint_attention_dim=16,
                                            axes_dims_rope=(4, 6, 6), dtype=torch.float32, device="cpu", operations=ops)
        x = torch.randn(3, 4, 1, 16, 20)
        kwargs = dict(context=torch.randn(3, 7, 16))
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return model.eval(), x, torch.tensor([0.7, 0.4, 0.2]), kwargs


def _worker(rank, world, comfy_path, store, family, results):
    import torch.distributed as dist
    _comfy(comfy_path)
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank, world_size=world)
    try:
        from powershard.image_parallel import ParallelContext, install, run_parallel
        torch.manual_seed(0)
        model, x, t, kwargs = _build(family)
        with torch.no_grad():
            reference = model(x, t, transformer_options={}, **kwargs)
            ctx = ParallelContext(world, rank)
            ctx.min_tokens = 4
            families = install(model, ctx)

            def call(x_, t_, **kw):
                return model(x_, t_, **kw)
            seq, stats_seq = run_parallel(call, ctx, "sequence", x, (x, t), dict(kwargs, transformer_options={}))
            batch, stats_batch = run_parallel(call, ctx, "batch", x, (x, t), dict(kwargs, transformer_options={}))
        if rank == 0:
            results.put(dict(families=families, seq=float((seq - reference).abs().max()),
                             batch=float((batch - reference).abs().max()), modes=(stats_seq["mode"], stats_batch["mode"]),
                             attention=stats_seq.get("attention_calls", 0)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["flux", "qwen_image"])
@pytest.mark.parametrize("world", [2, 3])
def test_sequence_and_batch_parallel_match_reference(request, tmp_path, family, world):
    comfy_path = request.config.getoption("--comfy")
    if comfy_path is None:
        pytest.skip("NOT_RUN: передайте --comfy")
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    procs = [ctx.Process(target=_worker, args=(r, world, comfy_path, str(tmp_path / "store"), family, results))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(300)
        assert p.exitcode == 0
    out = results.get(timeout=10)
    assert out["families"] and out["modes"] == ("sequence", "batch")
    assert out["attention"] > 0
    assert out["seq"] < 1e-4 and out["batch"] < 1e-5, out
