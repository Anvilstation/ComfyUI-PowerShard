"""Проверка Veda на настоящем GPU (по одному процессу на GPU, без ComfyUI-сессии).

  python scripts/probe_veda_cuda.py --gpu 0
  python scripts/probe_veda_cuda.py --gpu 0 --predictor /path/ComfyUI/models/veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors
  python scripts/probe_veda_cuda.py --gpu 0 --grid 37x24x42 --heads 7      # как один rank Ulysses на 6 GPU

1. self-test каждого бэкенда (triton-int8 / triton-fp16 / torch-sdpa) против FP32-эталона и замер;
2. полный путь движка на H3-подобной последовательности (text | video | audio):
   * разреженность 0% против плотной SDPA (проверка тайлинга/перестановок — ошибка должна быть ~1e-3);
   * разреженный режим (32 тайла на строку): время против плотной attention того же размера.
С --predictor используются его проекции и планы (первый слой), иначе случайные.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Layout:
    def __init__(self, text, grid, audio):
        t, h, w = grid
        video = t * h * w
        self.segments = [(0, text, "text"), (text, text + video, "video"), (text + video, text + video + audio, "audio")]
        self.signature = (text, t, 2 * h, 2 * w, audio)
        self.seq_len = text + video + audio
        self.position_ids = None


def timed(function, device, repeats=3):
    import torch
    function()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = function()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / repeats, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--predictor", default=None)
    parser.add_argument("--grid", default="37x24x42", help="T x H x W видео-токенов (после патча 1x2x2)")
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--budget", default="32")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from powershard.veda import backends, plans, selection, settings, tiling
    from powershard.veda.engine import VedaEngine
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    report = dict(gpu=torch.cuda.get_device_name(device), capability=torch.cuda.get_device_capability(device),
                  torch=torch.__version__, backends={})
    try:
        import triton
        report["triton"] = triton.__version__
    except ImportError as error:
        report["triton"] = f"нет: {error}"
    for name, cls in backends.CLASSES.items():
        try:
            backend = cls(device)
            error = backends.self_test(backend, device)
            report["backends"][name] = dict(self_test_error=error, bench_ms=1e3 * backends.benchmark(backend, device))
        except Exception as error:
            report["backends"][name] = dict(error=f"{type(error).__name__}: {error}"[:500])
    print(json.dumps(report["backends"], ensure_ascii=False, indent=1))

    grid = tuple(int(x) for x in args.grid.split("x"))
    layout = Layout(512, grid, 400)
    heads = args.heads
    if args.predictor:
        from powershard.veda.bundle import load_bundle
        bundle = load_bundle(args.predictor)
        heads = min(heads, bundle.num_heads)
    else:
        from powershard.veda.bundle import PredictorBundle
        plan = plans.TilePlan("probe", grid, [tiling.TileShape(2, 8, 8), tiling.TileShape(1, 8, 16)],
                              [[i % 2 for i in range(heads)]])
        proj = [torch.randn(heads, 384, 128).to(torch.bfloat16) * 0.05]
        bundle = PredictorBundle("random", 1, heads, 128, selection.Budget(tiles=32), selection.Budget(tiles=32),
                                 plans.PlanTable([plan]), proj, [p.clone() for p in proj], {})
    g = torch.Generator(device=device).manual_seed(0)
    q, k, v = (torch.randn(layout.seq_len, heads, 128, device=device, generator=g, dtype=torch.float16)
               for _ in range(3))
    scale = 128 ** -0.5
    qd, kd, vd = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    dense_s, dense = timed(lambda: F.scaled_dot_product_attention(qd, kd, vd, scale=scale)[0].transpose(0, 1), device)
    report.update(tokens=layout.seq_len, heads=heads, dense_sdpa_ms=1e3 * dense_s, full_path={})
    budget = selection.Budget(**settings.parse_sparsity(args.budget, "budget"))
    for name, info in report["backends"].items():
        if "error" in info:
            continue
        backend = backends.CLASSES[name](device)
        row = {}
        for label, generated in (("dense_check", selection.Budget(ratio=1.0)), ("sparse", budget)):
            engine = VedaEngine(bundle, generated, generated, backend, device)
            spec = engine.layout_spec(layout)
            plan = engine.plan_for(spec).plan
            seconds, out = timed(lambda: engine.attention(q, k, v, 0, spec, plan, torch.arange(heads), scale), device)
            row[label + "_ms"] = 1e3 * seconds
            if label == "dense_check":
                row["dense_check_max_error"] = float((out.float() - dense.float()).abs().max())
            else:
                row["attention_computed"] = engine.stats.report().get("attention_computed")
                row["speedup_vs_dense_sdpa"] = dense_s / seconds
        report["full_path"][name] = row
        torch.cuda.empty_cache()
    text = json.dumps(report, ensure_ascii=False, indent=1, default=str)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
