"""Real CPU/Gloo collectives, not a mock. CUDA/NCCL requires server tests."""
from datetime import timedelta
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from powershard.attention import (gather_rows, ulysses_heads_to_sequence,
                                 ulysses_sequence_to_heads, install_sequence,
                                 install_attention, configure_safe_qk)
from powershard.attention_contract import AttentionOptions, math_attention
from powershard.config import DistributedConfig, shard_bounds
from powershard.fp16_safe import apply_fp16_safe
from powershard.patch_config import H3PatchConfig


@pytest.fixture(scope="session")
def gloo_available(tmp_path_factory):
    """Report a real transport restriction as NOT_RUN, never as a passed mock."""
    directory = tmp_path_factory.mktemp("gloo-probe")
    try:
        dist.init_process_group("gloo", init_method=(directory/"store").as_uri(),
                                rank=0, world_size=1, timeout=timedelta(seconds=10))
    except RuntimeError as error:
        if "Operation not permitted" in str(error):
            pytest.skip("NOT_RUN: environment forbids Gloo TCP transport")
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def exchange_worker(rank, world, store, report):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank,
                            world_size=world, timeout=timedelta(seconds=60))
    try:
        errors = []
        for heads in (7, 56):
            total, dim = 2*world+1, 8  # uneven tokens AND uneven heads
            gen = torch.Generator().manual_seed(991)
            q, k = (torch.randn(total, heads, dim, generator=gen) for _ in range(2))
            v = torch.randn(total, heads, dim, generator=gen)*1e6
            a, b = shard_bounds(total, rank, world)
            (qg, kg, vg), padded_heads, _ = ulysses_heads_to_sequence((q[a:b], k[a:b], v[a:b]), total, heads)
            got = math_attention(qg[None], kg[None], vg[None], AttentionOptions(), 3, 5)[0]
            local = ulysses_sequence_to_heads(got, total, heads)
            reference = math_attention(q[None], k[None], v[None], AttentionOptions(), 3, 5)[0]
            torch.testing.assert_close(local, reference[a:b], rtol=2e-5, atol=.25)
            assert padded_heads == ((heads+world-1)//world)*world
            assert torch.isfinite(local).all()
            # Ceil-width partitioning can leave the last rank empty. The
            # equality assertion still checks its shape; an empty shard has
            # no numerical error to reduce.
            delta = (local-reference[a:b]).abs()
            errors.append(float(delta.max()) if delta.numel() else 0.)
        # Gather results must have distinct ownership, no mutable global cache.
        first = gather_rows(torch.full((b-a, 3), rank+100000., dtype=torch.float32), total)
        saved = first.clone()
        second = gather_rows(torch.full((b-a, 3), -rank-100000., dtype=torch.float32), total)
        torch.testing.assert_close(first, saved, atol=0, rtol=0)
        assert first.data_ptr() != second.data_ptr()
        if rank == 0:
            Path(report).write_text(json.dumps(dict(world_size=world, backend="gloo",
                status="PASS", heads=[7,56], v_magnitude=1e6, max_abs=errors)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [3, 4, 5, 6])
def test_real_ulysses_collectives(world, tmp_path, gloo_available):
    mp.spawn(exchange_worker, args=(world, str(tmp_path/"store"), str(tmp_path/"report.json")),
             nprocs=world, join=True)


def native_worker(rank, world, store, checkpoint, comfy_path, mode, provider):
    import sys
    torch.set_num_threads(1)
    sys.path.insert(0, comfy_path)
    sys.argv = ["powershard-test", "--cpu", "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from powershard.operations import Operations
    from powershard.checkpoint import Checkpoint
    from safetensors.torch import load_file
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank,
                            world_size=world, timeout=timedelta(seconds=90))
    try:
        ckpt = Checkpoint(checkpoint)
        net = MiniMaxH3Model(**ckpt.model_config(), dtype=torch.float16, device="cpu", operations=Operations)
        net.load_state_dict(load_file(checkpoint))
        net.eval().requires_grad_(False)
        config = DistributedConfig(sequence_mode=mode, sequence_comm_dtype="fp16", attention_backend=provider)
        install_attention(net, config)
        configure_safe_qk(net, ckpt)
        tracker = apply_fp16_safe(net, H3PatchConfig(enabled=True, debug_finite=True, mlp_chunk_tokens=3))
        # Force a finite residual >65504 before final assembly. Both runs use
        # the same modulation/heads; the old final FP16 gather fails this case.
        block = net.blocks[-1]
        native = block._ps_native_forward
        block._ps_native_forward = lambda *args, **kwargs: native(*args, **kwargs)*1e7
        gen = torch.Generator().manual_seed(17)
        video = torch.randn(1,24,2,8,8,generator=gen)
        audio = torch.randn(1,32,2,5,generator=gen)
        text = torch.full((1,7,ckpt.model_config()["text_dim"]), 1e5)
        with torch.no_grad():
            tracker.begin("cpu")
            ctx = net.preprocess_text_embeds(text)
            tracker.finish(ctx)
            tracker.begin("cpu")
            reference = net([video,audio],torch.tensor([700.]),ctx)
            tracker.finish(reference)
            install_sequence(net, config)
            tracker.begin("cpu")
            actual = net([video,audio],torch.tensor([700.]),ctx)
            tracker.finish(actual)
            for got, expected in zip(actual, reference):
                torch.testing.assert_close(got, expected, rtol=.01, atol=.03)
                assert got.dtype == torch.float32 and torch.isfinite(got).all()
            state = net.blocks[0].attn._ps_sequence
            assert state["enabled"] and state["ulysses"] == (mode=="ulysses")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [3,4,5,6])
@pytest.mark.parametrize("mode,provider", [("token","math"),("ulysses","math"),("ulysses","sdpa")])
def test_native_h3_sequence(world, mode, provider, h3_factory, request, tmp_path, gloo_available):
    from safetensors.torch import save_file
    net = h3_factory()
    # High V and final residual explicitly exercise the old FP16 cast failures.
    net.condition_proj.weight.fill_(1.)
    path = tmp_path/"h3.safetensors"
    save_file(net.state_dict(), str(path))
    mp.spawn(native_worker, args=(world, str(tmp_path/"store"), str(path),
                                  request.config.getoption("--comfy"), mode, provider),
             nprocs=world, join=True)


@pytest.mark.parametrize("world", [3,4,5,6,9])
def test_ulysses_tensor_permutation_reference(world, monkeypatch):
    """In-process collective reference: validates math/layout, NOT NCCL/Gloo."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    local = threading.local()
    barrier = threading.Barrier(world, timeout=30)
    slots = [None]*world
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: local.rank)
    def all_to_all(output, input, group=None):
        slots[local.rank] = input.clone()
        barrier.wait()
        output.copy_(torch.stack([slots[r][local.rank] for r in range(world)]))
        barrier.wait()
    def all_gather(output, input, group=None):
        slots[local.rank] = input.clone()
        barrier.wait()
        output.copy_(torch.cat(slots))
        barrier.wait()
    monkeypatch.setattr(dist, "all_to_all_single", all_to_all)
    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather)
    def worker(rank):
        local.rank = rank
        try:
            for heads in (1,7,56):
                total, dim = 2*world+1, 8
                gen = torch.Generator().manual_seed(991)
                q,k = (torch.randn(total,heads,dim,generator=gen) for _ in range(2))
                v = torch.randn(total,heads,dim,generator=gen)*1e6
                a,b = shard_bounds(total,rank,world)
                (qg,kg,vg),_,_ = ulysses_heads_to_sequence((q[a:b],k[a:b],v[a:b]),total,heads)
                out = math_attention(qg[None],kg[None],vg[None],AttentionOptions(),3,5)[0]
                actual = ulysses_sequence_to_heads(out,total,heads)
                reference = math_attention(q[None],k[None],v[None],AttentionOptions(),3,5)[0]
                torch.testing.assert_close(actual,reference[a:b],atol=.25,rtol=2e-5)
                first = gather_rows(v[a:b], total)
                saved = first.clone()
                gather_rows(-v[a:b], total)
                torch.testing.assert_close(first,saved,atol=0,rtol=0)
        except BaseException:
            barrier.abort()
            raise
    with ThreadPoolExecutor(max_workers=world) as executor:
        list(executor.map(worker,range(world)))


@pytest.mark.parametrize("world", [3,4,5,6])
@pytest.mark.parametrize("mode", ["token","ulysses"])
@pytest.mark.parametrize("communication", ["fp16","fp32"])
def test_native_h3_sequence_tensor_reference(world,mode,communication,h3_factory,monkeypatch):
    """Native H3 with emulated collective transport; NOT a distributed test.

    Check both intermediate high-V exchange and final >65504 FP32 residual.
    A legacy request for FP16 comm must not override the safe patch.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading
    torch.set_num_threads(1)
    config=DistributedConfig(sequence_mode=mode,sequence_comm_dtype=communication,attention_backend="sdpa")
    def create():
        net=h3_factory();net.condition_proj.weight.fill_(1.)
        install_attention(net,config)
        tracker=apply_fp16_safe(net,H3PatchConfig(enabled=True,debug_finite=True,mlp_chunk_tokens=3))
        block=net.blocks[-1];native=block._ps_native_forward
        block._ps_native_forward=lambda *args,**kwargs:native(*args,**kwargs)*1e7
        return net,tracker
    gen=torch.Generator().manual_seed(17)
    video=torch.randn(1,24,2,8,8,generator=gen);audio=torch.randn(1,32,2,5,generator=gen)
    text=torch.full((1,7,24),1e5)
    reference,tracker=create()
    with torch.no_grad():
        tracker.begin("cpu");context=reference.preprocess_text_embeds(text);tracker.finish(context)
        tracker.begin("cpu");expected=reference([video,audio],torch.tensor([700.]),context);tracker.finish(expected)
    # Preserve two other residual streams to detect reused-gather storage.
    models=[create() for _ in range(world)]
    local=threading.local();barrier=threading.Barrier(world,timeout=30);slots=[None]*world
    monkeypatch.setattr(dist,"get_world_size",lambda group=None:world)
    monkeypatch.setattr(dist,"get_rank",lambda group=None:local.rank)
    def all_to_all(output,input,group=None):
        slots[local.rank]=input.clone();barrier.wait()
        output.copy_(torch.stack([slots[r][local.rank] for r in range(world)]));barrier.wait()
    def all_gather(output,input,group=None):
        slots[local.rank]=input.clone();barrier.wait();output.copy_(torch.cat(slots));barrier.wait()
    def all_reduce(value,op=None,group=None):
        slots[local.rank]=value.clone();barrier.wait()
        value.copy_(torch.stack(slots).amax(0));barrier.wait()
    monkeypatch.setattr(dist,"all_to_all_single",all_to_all)
    monkeypatch.setattr(dist,"all_gather_into_tensor",all_gather)
    monkeypatch.setattr(dist,"all_reduce",all_reduce)
    for net,_ in models:install_sequence(net,config)
    def worker(rank):
        local.rank=rank;net,tracker=models[rank]
        try:
            with torch.no_grad():
                tracker.begin("cpu");actual=net([video,audio],torch.tensor([700.]),context);tracker.finish(actual)
                for got,ref in zip(actual,expected):torch.testing.assert_close(got,ref,rtol=.01,atol=.03)
                state=net.blocks[0].attn._ps_sequence
                assert state['enabled'] and state['ulysses']==(mode=='ulysses')
                assert state['effective_comm_dtype']=='torch.'+("float16" if communication=="fp16" else "float32")
                assert state['scale_collectives']==(len(net.blocks) if communication=="fp16" else 0)
        except BaseException:barrier.abort();raise
    with ThreadPoolExecutor(max_workers=world) as executor:list(executor.map(worker,range(world)))
