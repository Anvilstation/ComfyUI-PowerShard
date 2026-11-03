"""Veda (host, без torch): разбор настроек, метаданные предиктора, проверки ноды."""
import json
import struct
import sys
import types
import pytest


def write_predictor(path, layers=4, heads=8, head_dim=128, extra=None):
    metadata = {"format": "miowtion-veda-predictor-v1", "num_layers": str(layers), "num_heads": str(heads),
                "head_dim": str(head_dim), "target_budget_kind": "ratio", "target_budget_value": "0.1",
                "ref_budget_kind": "tiles", "ref_budget_value": "32", "plans": json.dumps({}), **(extra or {})}
    header = json.dumps({"__metadata__": metadata}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


H3 = dict(num_layers=4, num_attention_heads=8, attention_head_dim=128)


def test_settings_parsing():
    from powershard.veda import settings
    assert settings.parse_sparsity("90%", "x") == {"ratio": pytest.approx(0.1)}
    assert settings.parse_sparsity("24", "x") == {"tiles": 24.0}
    assert settings.parse_sparsity("trained", "x", {"tiles": 32.0}) == {"tiles": 32.0}
    assert settings.keeps_all(settings.parse_sparsity("0%", "x"))
    for bad in ("100%", "abc", "0", "-5"):
        with pytest.raises(ValueError):
            settings.parse_sparsity(bad, "x")
    assert settings.parse_index_list("0, 1, 47 - 49", "x") == [0, 1, 47, 48, 49]
    with pytest.raises(ValueError):
        settings.parse_index_list("a-b", "x")
    assert settings.step_index(0.75, [1.0, 0.75, 0.5, 0.0]) == 1
    assert settings.step_index(1.0, [1.0, 0.5, 0.0]) == 0


def test_predictor_metadata_and_spec(tmp_path):
    from powershard.veda import bundle
    from powershard.veda_nodes import build_spec
    path = write_predictor(tmp_path / "p.safetensors")
    metadata = bundle.read_header_metadata(str(path))
    assert bundle.geometry(metadata) == (4, 8, 128)
    assert bundle.trained_budgets(metadata) == ({"ratio": 0.1}, {"tiles": 32.0})
    spec = build_spec(str(path), H3, 6, "ulysses", "32", "trained", "0, 2-3", "0", "auto")
    assert spec["generated"] == {"tiles": 32.0} and spec["reference"] == {"tiles": 32.0}
    assert spec["dense_layers"] == [0, 2, 3] and spec["dense_steps"] == [0]
    json.dumps(spec)                                              # уходит в RPC как JSON
    with pytest.raises(ValueError, match="ulysses"):
        build_spec(str(path), H3, 6, "token", "32", "32", "", "", "auto")
    build_spec(str(path), H3, 1, "token", "32", "32", "", "", "auto")  # один GPU: режим не важен
    with pytest.raises(ValueError, match="обучен"):
        build_spec(str(path), dict(H3, num_attention_heads=40), 6, "ulysses", "32", "32", "", "", "auto")
    with pytest.raises(ValueError, match="блоки 0-3"):
        build_spec(str(path), H3, 6, "ulysses", "32", "32", "7", "", "auto")
    other = tmp_path / "x.safetensors"
    header = json.dumps({"__metadata__": {"format": "something"}}).encode()
    other.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(bundle.BundleError, match="не предиктор"):
        bundle.read_header_metadata(str(other))


def test_veda_node_registers_without_torch(monkeypatch, tmp_path):
    fake = types.ModuleType("folder_paths")
    fake.models_dir = str(tmp_path)
    fake.folder_names_and_paths = {}
    fake.get_filename_list = lambda folder: ["mine.safetensors"]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    from powershard.veda_nodes import VEDA_NODE_CLASS_MAPPINGS, KNOWN
    inputs = VEDA_NODE_CLASS_MAPPINGS["PowerShardH3Veda"].INPUT_TYPES()
    choices = inputs["required"]["predictor"][0]
    assert choices[0] == "mine.safetensors" and set(KNOWN) <= set(choices)
    assert inputs["optional"]["backend"][0] == ["auto", "triton-int8", "triton-fp16", "torch-sdpa"]
    assert (tmp_path / "veda").is_dir() and "veda" in fake.folder_names_and_paths
    with pytest.raises(ValueError, match="H3"):
        VEDA_NODE_CLASS_MAPPINGS["PowerShardH3Veda"]().apply(object(), "mine.safetensors")
