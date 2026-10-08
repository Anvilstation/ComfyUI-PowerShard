"""Настоящие collectives (gloo, CPU): sequence token/Ulysses и FSDP2 WanBackend против native Wan.

Запуск: python -m pytest tests/test_wan_distributed.py --comfy /path/to/ComfyUI
Если среда запрещает сокеты/gloo или FSDP2 на CPU — тест помечается skip с причиной,
а не PASS. CUDA/NCCL на V100 проверяется scripts/probe_wan_cuda.py.
"""
import json
import os
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")


def _worker(rank, world, folder, comfy, mode, comm, fsdp, case="i2v"):
    import sys
    import traceback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    result = dict(rank=rank)
    try:
        import torch.distributed as dist
        from wan_reference import import_comfy, native_model, powershard_model
        wan = import_comfy(comfy)
        dist.init_process_group("gloo", rank=rank, world_size=world, init_method=Path(folder, "store").as_uri())
        from powershard.config import DistributedConfig
        from powershard.wan_config import WanOptions
        from wan_reference import variant_call, our_options
        config, build = variant_call(case)
        native = native_model(wan, config, seed=5)
        dconfig = DistributedConfig(attention_backend="math", sequence_mode=mode, sequence_comm_dtype=comm)
        options = WanOptions(mlp_chunk_mode="manual", mlp_chunk_tokens=5)
        with torch.no_grad():
            args, kwargs, native_kwargs, native_options = build(native)
            expected = native(*args, transformer_options=dict(native_options), **native_kwargs)
            if fsdp:
                from powershard.wan_backend import WanBackend
                path = Path(folder, "ckpt.safetensors")
                backend = WanBackend(str(path), dconfig, torch.device("cpu"), options, None, "main", ())
                actual, metrics = backend.call("forward", args, dict(transformer_options={}, **kwargs))
                result["metrics"] = dict(sequence=metrics.get("sequence_communication"), duplicated=metrics["duplicated_compute"],
                                         shards=len(backend.shards))
            else:
                ours, tracker = powershard_model(wan, config, native, options, dconfig)
                tracker.begin("cpu")
                actual = ours("forward", args, dict(kwargs, transformer_options=our_options(native_options)))
                tracker.finish(actual)
                result["sequence"] = {k: v for k, v in ours.network._ps_sequence.items() if isinstance(v, (int, float, str, bool))}
        result["max_abs"] = float((actual - expected).abs().max())
        result["scale"] = float(expected.abs().max())
        result["status"] = "PASS"
        dist.destroy_process_group()
    except Exception as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}", trace=traceback.format_exc()[-3000:])
    Path(folder, f"rank{rank}.json").write_text(json.dumps(result))


def run(comfy, tmp_path, world, mode, comm, fsdp=False, case="i2v"):
    import torch.multiprocessing as mp
    if fsdp:
        from wan_reference import import_comfy, tiny_config, native_model, save_native
        wan = import_comfy(comfy)
        save_native(native_model(wan, tiny_config(model_type="i2v", in_dim=36), seed=5), tmp_path / "ckpt.safetensors")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_worker, args=(world, str(tmp_path), comfy, mode, comm, fsdp, case), nprocs=world, join=True)
    results = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(world)]
    errors = [r for r in results if r["status"] != "PASS"]
    if errors:
        text = errors[0]["error"]
        if any(word in text for word in ("Operation not permitted", "gloo", "Gloo", "not supported", "NotImplemented")):
            pytest.skip("NOT_RUN: среда не поддерживает " + text[:300])
        raise AssertionError(errors[0]["trace"])
    for r in results:
        assert r["max_abs"] <= 3e-2 + 3e-2 * r["scale"], r
    return results


@pytest.fixture
def comfy(request):
    path = request.config.getoption("--comfy")
    if path is None:
        pytest.skip("NOT_RUN: передайте --comfy")
    return path


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("mode", ["token", "ulysses"])
def test_sequence_parallel_matches_native(comfy, tmp_path, world, mode):
    results = run(comfy, tmp_path, world, mode, "fp32")
    assert results[0]["sequence"]["enabled"]
    if mode == "ulysses":
        assert results[0]["sequence"]["padded_heads"] % world == 0


@pytest.mark.parametrize("mode", ["token", "ulysses"])
def test_fp16_wire(comfy, tmp_path, mode):
    run(comfy, tmp_path, 3, mode, "fp16")


@pytest.mark.parametrize("mode", ["token", "ulysses"])
def test_fsdp2_backend_cpu(comfy, tmp_path, mode):
    results = run(comfy, tmp_path, 3, mode, "fp32", fsdp=True)
    assert not results[0]["metrics"]["duplicated"]


@pytest.mark.parametrize("case", ["vace", "s2v", "animate", "uni3c", "humo", "scail", "scail2", "wandancer",
                                  "wandancer30", "animate2", "multitalk1", "multitalk2"])
@pytest.mark.parametrize("mode", ["token", "ulysses"])
def test_variants_sequence_parallel(comfy, tmp_path, case, mode):
    """3 ранка: границы shard проходят внутри кадров/аудио-сегментов/VACE, Uni3C, pose-веток и карт говорящих."""
    results = run(comfy, tmp_path, 3, mode, "fp32", case=case)
    assert results[0]["sequence"]["enabled"]


@pytest.mark.parametrize("case", ["animate2", "multitalk2", "humo"])
@pytest.mark.parametrize("mode", ["token", "ulysses"])
def test_new_variants_fp16_wire(comfy, tmp_path, case, mode):
    """FP16 wire: общий V scale генерации и pose branch, x_ref_attn_map из нормализованных q/k."""
    run(comfy, tmp_path, 4, mode, "fp16", case=case)
