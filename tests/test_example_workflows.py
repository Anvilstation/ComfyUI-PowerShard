"""example_workflows/*.json: актуальны относительно генератора, связи и виджеты согласованы с INPUT_TYPES."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "example_workflows").glob("*.json"))


def test_files_match_generator():
    assert len(FILES) >= 7
    subprocess.run([sys.executable, str(ROOT / "scripts/build_example_workflows.py"), "--check"], check=True)


def test_links_and_widgets_consistent():
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_example_workflows import SPECS, SEED_CONTROL
    for path in FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert not path.name.endswith(".api.json")
        nodes = {n["id"]: n for n in data["nodes"]}
        assert data["last_node_id"] == max(nodes) and data["last_link_id"] == len(data["links"])
        for link_id, source, slot, target, target_slot, kind in data["links"]:
            assert nodes[target]["inputs"][target_slot]["link"] == link_id, path.name
            assert link_id in nodes[source]["outputs"][slot]["links"], path.name
            assert nodes[source]["outputs"][slot]["type"] == kind == nodes[target]["inputs"][target_slot]["type"]
        for node in data["nodes"]:
            inputs, outputs = SPECS[node["type"]]
            widgets = [name for name, kind in inputs if kind == "W"]
            extra = 1 if node["type"] in SEED_CONTROL else 0
            assert len(node["widgets_values"]) == len(widgets) + extra, (path.name, node["type"])
            assert [o["type"] for o in node["outputs"]] == outputs
            if node["type"].startswith("PowerShard"):
                required = [name for name, kind in inputs if kind != "W"]
                linked = {s["name"] for s in node["inputs"] if s["link"] is not None}
                assert {"config", "model", "samples"} & set(required) <= linked, (path.name, node["type"])
