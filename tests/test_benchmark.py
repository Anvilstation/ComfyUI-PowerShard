"""Проверка границ измерений, не доказательство запуска GPU или ComfyUI UI."""
import asyncio
import json
import types
import pytest
from powershard import benchmark


@pytest.mark.parametrize("pending", [False, True])
def test_execution_measurement(tmp_path, monkeypatch, pending):
    monkeypatch.setattr(benchmark, "cuda_snapshot", lambda reset=False: [])
    monkeypatch.setattr(benchmark.GPUInventory, "start", lambda self: None)
    async def get_output(prompt_id, node_id, obj, data):
        return (), {}, False, pending
    class Executor:
        async def execute_async(self, prompt, prompt_id):
            self.status_messages = [("execution_cached", {"nodes": ["2"]})]
            self.success = True
            await module.get_output_data(prompt_id, "1", object(), {})
            return "result"
    module = types.SimpleNamespace(get_output_data=get_output, PromptExecutor=Executor)
    assert benchmark.install(module, tmp_path)
    assert not benchmark.install(module, tmp_path)
    graph = {"1": {"class_type": "VAEDecodeAudio"}, "2": {"class_type": "CLIPTextEncode","inputs":{"text":"private test prompt"}}}
    assert asyncio.run(module.PromptExecutor().execute_async(graph, "../untrusted-id")) == "result"
    report = json.loads(next(tmp_path.glob("prompt-*.json")).read_text())
    assert report["cached_nodes"] == ["2"]
    assert len(report["nodes"]) == 1
    assert report["nodes"][0]["stage"] == "audio_vae_decode_including_lazy_load"
    assert report["nodes"][0]["status"] == ("ASYNC_OR_SUBGRAPH_NOT_TIMED" if pending else "PASS")
    assert report["execution_s"] >= report["nodes"][0]["wall_s"] >= 0
    assert "private test prompt" not in next(tmp_path.glob("prompt-*.json")).read_text()
    assert report["graph"]["2"]["inputs"]["text"]["redacted"]
