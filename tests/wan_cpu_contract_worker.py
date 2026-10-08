"""Только тестовая точка входа: один CPU worker Wan, без FSDP/NCCL. Протокол как у wan_worker."""
import json
from pathlib import Path
import sys
import traceback

settings = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, settings["comfy_path"])
sys.argv = ["wan-test-worker", "--cpu", "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
protocol = sys.stdout
sys.stdout = sys.stderr


def respond(value):
    protocol.write(json.dumps(value) + "\n")
    protocol.flush()


try:
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    from safetensors.torch import load_file
    from powershard.attention_policy import AttentionDispatcher
    from powershard.config import DistributedConfig
    from powershard.wan_config import WanCheckpoint, WanOptions, model_kwargs
    from powershard.wan_model import (WanEntrypoint, build_network, make_fp32_parameters, configure_network,
                                      qk_norm_names, apply_qk_scales)
    from powershard.wire import read_payload, write_payload
    role = settings["role_options"]
    options = WanOptions(**role["options"])
    config = DistributedConfig(**settings["config"])
    experts = {}
    for slot, path in role["experts"].items():
        ckpt = WanCheckpoint(path)
        net = build_network(model_kwargs(ckpt.model_config()))
        with torch.device("meta"):
            make_fp32_parameters(net)
        net.to_empty(device="cpu")
        weights = {k[len(ckpt.prefix):]: v for k, v in load_file(path).items()}
        with torch.no_grad():
            for name, p in net.named_parameters():
                p.copy_(weights[name].to(p.dtype))
        tracker, _ = configure_network(net, config, options, AttentionDispatcher(config))
        params = dict(net.named_parameters())
        apply_qk_scales(net, {n: float(params[n].abs().max()) for n in qk_norm_names(net)})
        experts[slot] = (WanEntrypoint(net), tracker)
    respond({"sequence": 0, "preflight": {"test_mode": "CPU_SUBPROCESS_NO_FSDP", "patch_fingerprint": settings["patch_fingerprint"],
                                          "attention": {"policy": {"fingerprint": "cpu-test"}}, "wan_fingerprints": {}}})
    expected, stage_cache, stage_dir = 1, {}, None
    for line in sys.stdin:
        req = json.loads(line)
        if req["command"] == "shutdown":
            break
        assert req["sequence"] == expected
        expected += 1
        if req["command"] in ("idle", "end_run"):
            stage_cache.clear()
            key = "phase_offload" if req["command"] == "idle" else "end_run"
            respond({"sequence": req["sequence"], key: {"after": {"allocated": 0}, "test_mode": "CPU_NO_FSDP"}})
            continue
        if req.get("stage") != stage_dir:
            stage_cache.clear()
            stage_dir = req.get("stage")
        payload = read_payload(req["input"], base_directory=stage_dir or None, stage_cache=stage_cache)
        kwargs = dict(payload["kwargs"])
        slot = kwargs.pop("_powershard_expert")
        root, tracker = experts[slot]
        with torch.inference_mode(False), torch.no_grad():
            tracker.begin("cpu")
            result = root(req["command"], payload["args"], kwargs)
            tracker.finish(result)
        write_payload(req["output"], result)
        respond({"sequence": req["sequence"], "metrics": {"test_mode": "CPU_SUBPROCESS_NO_FSDP", "expert": slot,
                                                          "memory": {"allocated": 0}}})
except BaseException:
    respond({"error": traceback.format_exc()})
    raise
