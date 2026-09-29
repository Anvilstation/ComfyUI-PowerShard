"""Только тестовая точка входа: один CPU worker, tiny H3, никакой имитации CUDA/FSDP."""
import json
from pathlib import Path
import sys
import traceback

settings=json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,settings["comfy_path"])
sys.argv=["test-worker","--cpu","--disable-dynamic-vram","--use-pytorch-cross-attention"]
protocol=sys.stdout;sys.stdout=sys.stderr
def respond(obj):protocol.write(json.dumps(obj)+"\n");protocol.flush()
try:
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from powershard.checkpoint import Checkpoint
    from powershard.operations import Operations
    from powershard.attention import install_attention
    from powershard.config import DistributedConfig
    from powershard.patch_config import H3PatchConfig
    from powershard.fp16_safe import apply_fp16_safe
    from powershard.fsdp_backend import Entrypoint
    from powershard.wire import read_payload,write_payload
    from safetensors.torch import load_file
    ck=Checkpoint(settings["checkpoint"])
    with torch.device("meta"):net=MiniMaxH3Model(**ck.model_config(),dtype=torch.float16,device="meta",operations=Operations)
    net.to_empty(device="cpu").eval().requires_grad_(False)
    net.load_state_dict(load_file(settings["checkpoint"]))
    install_attention(net,DistributedConfig())
    patch=H3PatchConfig(**settings["patch"]);tracker=apply_fp16_safe(net,patch);root=Entrypoint(net)
    from powershard.spectrum import SpectrumEngine,install_spectrum
    from powershard.spectrum_config import SpectrumConfig
    spectrum=SpectrumEngine(SpectrumConfig(**settings.get("role_options",{}).get("spectrum",{})))
    install_spectrum(net,spectrum)
    respond({"sequence":0,"test_mode":"CPU_SUBPROCESS_NO_FSDP","patch":patch.to_dict()})
    expected=1
    for line in sys.stdin:
        req=json.loads(line)
        if req["command"]=="shutdown":break
        assert req["sequence"]==expected;expected+=1
        if req["command"]=="end_run":
            respond({"sequence":req["sequence"],"spectrum_end_run":spectrum.report()});spectrum.clear();continue
        payload=read_payload(req["input"], base_directory=req.get("stage") or None)
        metadata=payload["kwargs"].pop("_powershard_spectrum",None)
        if req["command"]=="forward":spectrum.begin(metadata)
        with torch.inference_mode(False),torch.no_grad():
            tracker.begin("cpu")
            result=root(req["command"],payload["args"],payload["kwargs"])
            tracker.finish(result)
        write_payload(req["output"],result)
        respond({"sequence":req["sequence"],"metrics":{"test_mode":"CPU_SUBPROCESS_NO_FSDP","spectrum":spectrum.report()}})
except BaseException:
    respond({"error":traceback.format_exc()})
    raise
