"""Дельты LoRA / LoHa / LoKr по строкам shard == полная дельта (как comfy.weight_adapter). Нужны torch и safetensors.

  python -m pytest tests/test_lora_torch.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def merger_for(tmp_path, tensors, shape, name="blocks.0.self_attn.q.weight"):
    from safetensors.torch import save_file
    from powershard.wan_lora import LoraMerger
    path = tmp_path / "lora.safetensors"
    save_file(tensors, str(path))
    return LoraMerger([dict(path=str(path), strength=0.7)], {name: {"shape": shape}}), name


@pytest.mark.parametrize("bounds", [(0, 24), (5, 17), (23, 24)])
def test_lokr_rows_match_full_kron(tmp_path, bounds):
    g = torch.Generator().manual_seed(0)
    w1 = torch.randn(4, 2, generator=g)
    w2a, w2b = torch.randn(6, 3, generator=g), torch.randn(3, 8, generator=g)
    p = "diffusion_model.blocks.0.self_attn.q."
    merger, name = merger_for(tmp_path, {p + "lokr_w1": w1, p + "lokr_w2_a": w2a, p + "lokr_w2_b": w2b,
                                         p + "alpha": torch.tensor(1.5)}, [24, 16])
    full = torch.kron(w1, w2a @ w2b) * (1.5 / 3) * 0.7
    a, b = bounds
    torch.testing.assert_close(merger.delta(name, a, b, [24, 16]), full[a:b])


def test_loha_rows_match_full(tmp_path):
    g = torch.Generator().manual_seed(1)
    t = {k: torch.randn(*s, generator=g) for k, s in
         (("hada_w1_a", (12, 2)), ("hada_w1_b", (2, 10)), ("hada_w2_a", (12, 2)), ("hada_w2_b", (2, 10)))}
    p = "diffusion_model.blocks.0.self_attn.q."
    merger, name = merger_for(tmp_path, {p + k: v for k, v in t.items()} | {p + "alpha": torch.tensor(4.)}, [12, 10])
    full = (t["hada_w1_a"] @ t["hada_w1_b"]) * (t["hada_w2_a"] @ t["hada_w2_b"]) * (4. / 2) * 0.7
    torch.testing.assert_close(merger.delta(name, 3, 9, [12, 10]), full[3:9])


def test_peft_default_lora_rows(tmp_path):
    g = torch.Generator().manual_seed(2)
    down, up = torch.randn(4, 16, generator=g), torch.randn(24, 4, generator=g)
    p = "diffusion_model.blocks.0.self_attn.q."
    merger, name = merger_for(tmp_path, {p + "lora_A.default.weight": down, p + "lora_B.default.weight": up}, [24, 16])
    torch.testing.assert_close(merger.delta(name, 2, 11, [24, 16]), (up @ down)[2:11] * 0.7)
