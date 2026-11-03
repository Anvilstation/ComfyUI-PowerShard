"""Veda (torch, CPU): движок с подмножеством голов, бэкенд torch-sdpa против FP32-эталона.

  python -m pytest tests/test_veda_torch.py
GPU-бэкенды (triton-int8/fp16) проверяет scripts/probe_veda_cuda.py.
"""
import types
import pytest

torch = pytest.importorskip("torch")


class FakeLayout:
    """Минимальный PackedLayout H3: text | video (T,H,W после патча 1x2x2) | audio."""

    def __init__(self, text=40, grid=(3, 8, 12), audio=30):
        t, h, w = grid
        video = t * h * w
        self.segments = [(0, text, "text"), (text, text + video, "video"), (text + video, text + video + audio, "audio")]
        self.signature = (text, t, h * 2, w * 2, audio)
        self.seq_len = text + video + audio
        self.position_ids = None


def fake_bundle(layers=2, heads=4, dim=128, grid=(3, 8, 12)):
    from powershard.veda import plans, tiling
    from powershard.veda.bundle import PredictorBundle
    from powershard.veda.selection import Budget
    g = torch.Generator().manual_seed(0)
    plan = plans.TilePlan("t", grid, [tiling.TileShape(1, 8, 16), tiling.TileShape(2, 8, 8)],
                          [[i % 2 for i in range(heads)] for _ in range(layers)])
    proj = [torch.randn(heads, 3 * dim, dim, generator=g).to(torch.bfloat16) * 0.05 for _ in range(layers)]
    return PredictorBundle("fake.safetensors", layers, heads, dim, Budget(tiles=4), Budget(tiles=4),
                           plans.PlanTable([plan]), proj, [p.clone() for p in proj], {})


class ReferenceBackend:
    name = "reference"

    def attend(self, q, k, v, mask, layout, softmax_scale):
        from powershard.veda.reference import block_sparse_attention
        return block_sparse_attention(q, k, v, mask, layout, softmax_scale).to(q.dtype)


def qkv(seq, heads, seed=1, dtype=torch.float16):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(seq, heads, 128, generator=g) * 1.2).to(dtype) for _ in range(3)]


def engine(backend, generated, reference=None):
    from powershard.veda.engine import VedaEngine
    from powershard.veda.selection import Budget
    bundle = fake_bundle()
    return VedaEngine(bundle, Budget(**generated), Budget(**(reference or generated)), backend, torch.device("cpu"))


def test_torch_sdpa_backend_self_test_cpu():
    from powershard.veda.backends import TorchSDPA, self_test
    assert self_test(TorchSDPA(torch.device("cpu")), torch.device("cpu")) < 0.02


def test_keep_all_equals_dense_attention():
    from powershard.veda.backends import TorchSDPA
    layout = FakeLayout()
    eng = engine(TorchSDPA(torch.device("cpu")), dict(ratio=1.0))
    q, k, v = qkv(layout.seq_len, 4)
    spec = eng.layout_spec(layout)
    out = eng.attention(q, k, v, 0, spec, eng.plan_for(spec).plan, torch.arange(4), 128 ** -0.5)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * 128 ** -0.5
    dense = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    torch.testing.assert_close(out.float(), dense, atol=3e-3, rtol=3e-3)


def test_sparse_pipeline_matches_reference_and_head_subset():
    from powershard.veda.backends import TorchSDPA
    layout = FakeLayout()
    q, k, v = qkv(layout.seq_len, 4, seed=3)
    ref = engine(ReferenceBackend(), dict(tiles=3))
    fast = engine(TorchSDPA(torch.device("cpu")), dict(tiles=3))
    spec = ref.layout_spec(layout)
    plan = ref.plan_for(spec).plan
    want = ref.attention(q, k, v, 1, spec, plan, torch.arange(4), 0.1)
    got = fast.attention(q, k, v, 1, spec, plan, torch.arange(4), 0.1)
    torch.testing.assert_close(got.float(), want.float(), atol=3e-3, rtol=3e-3)
    # Ulysses: rank видит головы 2..3 и одну паддинг-голову (глобальный индекс 4 >= 4 голов модели)
    part = [torch.cat([t[:, 2:4], torch.zeros_like(t[:, :1])], 1) for t in (q, k, v)]
    sub = fast.attention(*part, 1, spec, plan, torch.tensor([2, 3, 4]), 0.1)
    torch.testing.assert_close(sub[:, :2].float(), got[:, 2:4].float(), atol=1e-3, rtol=1e-3)
    assert sub[:, 2].abs().max() == 0
    assert 0 < fast.stats.report()["video_tiles_kept"] < 1


def test_runtime_declines_token_mode_and_dense_layers():
    from powershard.veda.runtime import VedaRuntime
    def attn():
        return types.SimpleNamespace(heads=4, head_dim=128, _ps_attention=lambda *a, **k: "dense",
                                     forward=lambda x, rope_freqs=None, transformer_options=None: x)
    net = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn=attn()) for _ in range(2)])
    runtime = VedaRuntime(net, torch.device("cpu"))
    layout = FakeLayout()
    q = torch.zeros(1, layout.seq_len, 4, 128)
    assert runtime.decline_reason(q, q, q, None) == "off"
    runtime.active, runtime.engine = True, engine(ReferenceBackend(), dict(tiles=3))
    runtime.config = (None, frozenset({1}), frozenset())
    net.blocks[0].attn.forward(q, transformer_options={"block_index": 0, "minimax_h3_layout": layout})
    assert runtime.decline_reason(q, q, q, None) is None
    local = q[:, :100]
    assert "token" in runtime.decline_reason(local, q, q, None)
    runtime.context["block_index"] = 1
    assert runtime.decline_reason(q, q, q, None) == "full-attention слой"
    # Диспетчер исходного attention остаётся доступным через обёртку (effective/report).
    assert net.blocks[0].attn._ps_attention.inner(q, q, q) == "dense"


def test_explicit_backend_is_not_blocked_by_self_test(monkeypatch):
    """Явный выбор: только компиляция/первый запуск; auto: self-test, но провал кандидата не останавливает выбор."""
    from powershard.veda import backends
    calls = []
    monkeypatch.setattr(backends, "self_test", lambda *a: calls.append("self_test") or 0.0)
    created, attempts, timings = backends.evaluate(torch.device("cpu"), "torch-sdpa")
    assert "torch-sdpa" in created and calls == [] and attempts["torch-sdpa"].startswith("ok")
    monkeypatch.setattr(backends, "self_test", lambda b, d: (_ for _ in ()).throw(backends.BackendUnavailable("x")))
    created, attempts, timings = backends.evaluate(torch.device("cpu"), "auto")
    assert not created and all(t == float("inf") for t in timings)   # отказ — в отчёте, не исключение


def test_prepare_reports_every_failure(tmp_path, monkeypatch):
    from powershard.veda import runtime
    monkeypatch.setattr(runtime, "load_bundle_cached", lambda path: (_ for _ in ()).throw(OSError("нет файла")))
    with pytest.raises(RuntimeError, match="rank 0.*нет файла"):
        runtime.prepare_collectively(torch.device("cpu"), str(tmp_path / "p"), "torch-sdpa", (1, 1, 128))
