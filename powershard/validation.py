"""Настоящий FSDP2 smoke: frozen FP16/INT8, три ранга, повторный forward без backward."""
import gc
import tempfile
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy
from .config import shard_bounds
from .fsdp_backend import assert_sharded, memory, sync, load_local
from .checkpoint import Checkpoint
from .operations import Linear, Int8Linear


class ProbeNet(nn.Module):
    def __init__(self, quant=False):
        super().__init__()
        layer = Linear(256, 13, bias=False, device="meta", dtype=torch.float16)
        if quant:
            layer = Int8Linear(layer, {"convrot": True, "group_size": 256}, rows=4)
        self.blocks = nn.ModuleList([nn.Sequential(layer), nn.Sequential(Linear(13, 7, device="meta", dtype=torch.float16))])
        self.tail = Linear(7, 5, device="meta", dtype=torch.float32)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.tail(x)


class ProbeRoot(nn.Module):
    def __init__(self, net):
        super().__init__(); self.network = net
    def forward(self, x):
        return self.network(x)


def distributed_probe(device, cpu_offload=False, pin_memory=True, prefetch_blocks=0, expected_world=None, managed_pool=None):
    rank = dist.get_rank()
    world = dist.get_world_size()
    if expected_world is not None and world != expected_world:
        raise ValueError("Число ranks отличается от выбранного набора GPU")
    policy = CPUOffloadPolicy(pin_memory=pin_memory) if cpu_offload else OffloadPolicy()
    result = {}
    for quant in (False, True):
        gen = torch.Generator().manual_seed(42)
        cpu = ProbeNet(quant).to_empty(device="cpu").eval().requires_grad_(False)
        for name, p in cpu.named_parameters():
            if p.dtype == torch.int8:
                p.copy_(torch.randint(-8, 8, p.shape, generator=gen, dtype=torch.int8))
            elif name.endswith("weight_scale"):
                p.fill_(.002)
            else:
                p.copy_(torch.randn(p.shape, generator=gen).mul_(.025).to(p.dtype))
        state = cpu.state_dict()
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory(prefix="powershard-probe-") as tmp:
            path = Path(tmp)/"tiny.safetensors"
            save_file(state, str(path))
            root = ProbeRoot(ProbeNet(quant)).eval().requires_grad_(False)
            mesh = init_device_mesh(device.type, (world,))
            for b in root.network.blocks:
                fully_shard(b, mesh=mesh, reshard_after_forward=True,
                            mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False), offload_policy=policy)
            fully_shard(root, mesh=mesh, reshard_after_forward=True,
                        mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False), offload_policy=policy)
            chain = list(root.network.blocks)
            for i,b in enumerate(chain):b.set_modules_to_forward_prefetch(chain[i+1:i+1+prefetch_blocks])
            # quant_map нужен только для квантованных embedding-исключений
            # (load_local деквантует floating-параметры из I8-источника).
            # Веса Int8Linear остаются сырыми I8 и в map не входят.
            evidence = load_local(root, Checkpoint(path), device, rank, world, cpu_offload=cpu_offload, quant_map={}, managed_pool=managed_pool)
        # Реконструкция разрешена только для tiny test, не для генератора H3.
        for name, p in root.named_parameters():
            target = state[name.removeprefix("network.")].to(device)
            # CPU-resident shards can't run a CPU collective in NCCL. Tiny-only
            # move to GPU for reference reconstruction, not a production loader.
            reconstructed = p.to(device).full_tensor()
            torch.testing.assert_close(reconstructed, target, rtol=0, atol=0)
        pieces = [None]*world
        dist.all_gather_object(pieces, evidence)
        for idx in range(len(evidence)):
            intervals = [r[idx]["rows"] for r in pieces]
            assert intervals[0][0] == 0 and intervals[-1][1] == evidence[idx]["shape"][0]
            assert all(intervals[i][1] == intervals[i+1][0] for i in range(world-1))
        x = torch.randn((3,256), generator=gen).half()
        with torch.no_grad():
            reference = cpu(x).to(device)
        memories, errors = [], []
        for step in range(5):
            with torch.inference_mode(False), torch.no_grad():
                y = root(x.to(device))
            sync(device)
            assert_sharded(root, cpu_offload if device.type=="cuda" else None)
            if managed_pool is not None:
                managed_pool.assert_parameters(root)
            torch.testing.assert_close(y, reference, atol=2e-3, rtol=3e-3)
            errors.append((y-reference).abs().max().item())
            memories.append(memory(device))
        if device.type == "cuda" and memories[-1]["allocated"] > memories[1]["allocated"] + 1024**2:
            raise AssertionError("Память растёт между forward без backward")
        result["int8" if quant else "fp16"] = {"status": "PASS", "steps": 5, "max_abs": max(errors),
                                                   "shards": evidence, "memory": memories,
                                                   "mode": "eval+no_grad", "reconstruction": "PASS"}
        result["int8" if quant else "fp16"]["cpu_offload"] = cpu_offload
        result["int8" if quant else "fp16"]["pin_memory"] = pin_memory
        result["int8" if quant else "fp16"]["world_size"] = world
        del root, cpu, y, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def native_h3_probe(device, config, folder, managed_pool=None):
    """Одинаковая настоящая H3 topology, tiny random weights; оба storage modes.

    Full CPU reference разрешена ТОЛЬКО для этой малой тестовой модели.
    Production H3Backend никогда не вызывает эту функцию.
    """
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from .operations import Operations, install_int8
    from .attention import install_attention, install_sequence
    from .attention_policy import AttentionDispatcher
    from dataclasses import replace
    from .fp16_safe import apply_fp16_safe
    from .patch_config import H3PatchConfig
    from .fsdp_backend import Entrypoint, wrap_fsdp
    from safetensors.torch import save_file
    import json
    rank, world = dist.get_rank(), dist.get_world_size()
    shape = dict(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=7,
        attention_head_dim=8, ffn_hidden_size=512, text_dim=256, time_embed_dim=4,
        adaln_curve_grid=17, rope_inv_freq_len=1, latents_dim=24, audio_latents_dim=32)
    patch = H3PatchConfig(enabled=True, debug_finite=True, mlp_chunk_tokens=3)
    output = {}
    for quant in (False, True):
        with torch.device('meta'):
            reference = MiniMaxH3Model(**shape,dtype=torch.float16,device='meta',operations=Operations)
            mapping = {name:dict(convrot=True,group_size=256) for name,m in reference.named_modules()
                       if quant and isinstance(m,Linear) and m.in_features%256==0 and m.weight.dtype==torch.float16}
            install_int8(reference,mapping,64)
        reference.to_empty(device='cpu').eval().requires_grad_(False)
        gen=torch.Generator().manual_seed(17)
        for name,param in reference.named_parameters():
            if param.dtype==torch.int8:param.copy_(torch.randint(-8,8,param.shape,generator=gen,dtype=torch.int8))
            elif name.endswith('weight_scale'):param.fill_(.001)
            elif 'norm' in name:param.fill_(1.)
            else:param.copy_(torch.randn(param.shape,generator=gen).mul_(.01).to(param.dtype))
        reference.adaln_t_table.fill_(.1);reference.rope.inv_freq.fill_(.01)
        # condition input заведомо превышает half limit, но истинный FP32 конечен.
        text=torch.full((1,7,256),100000.)
        video=torch.randn((1,24,2,8,8),generator=gen)
        audio=torch.randn((1,32,2,5),generator=gen)
        kwargs={'transformer_options':{'sample_sigmas':torch.tensor([1.,.5,0.])},'minimax_payload':{'seed':17}}
        state=dict(reference.state_dict())
        for name in mapping:
            raw=json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}).encode()
            state[name+'.comfy_quant']=torch.tensor(list(raw),dtype=torch.uint8)
        ckpath=Path(folder,f'tiny-h3-{quant}.safetensors')
        if rank==0:save_file(state,str(ckpath))
        dist.barrier()
        # CPU reference cannot call a CUDA-only custom provider.
        reference_config=replace(config,attention_backend="math")
        install_attention(reference,reference_config);tracker=apply_fp16_safe(reference,patch)
        with torch.no_grad():
            tracker.begin('cpu');ctx_ref=reference.preprocess_text_embeds(text);tracker.finish(ctx_ref)
            tracker.begin('cpu');y_ref=reference([video,audio],torch.tensor([700.]),ctx_ref,**kwargs);tracker.finish(y_ref)
        with torch.device('meta'):
            net=MiniMaxH3Model(**shape,dtype=torch.float16,device='meta',operations=Operations)
            install_int8(net,mapping,64);root=Entrypoint(net)
        root.eval().requires_grad_(False)
        # This diagnostic is deliberately math/SDPA only; custom kernel
        # certification and real-checkpoint acceptance run in Session.
        install_attention(net,config,AttentionDispatcher(config));tracker=apply_fp16_safe(net,patch)
        install_sequence(net,config)
        mesh=init_device_mesh(device.type,(world,))
        units=wrap_fsdp(root,mesh,config)
        # mapping тут — только имена Int8Linear'ов; quant_map для load_local
        # нужен только для embedding-исключений, у tiny-H3 их нет.
        evidence=load_local(root,Checkpoint(ckpath),device,rank,world,cpu_offload=config.cpu_offload,quant_map={},managed_pool=managed_pool)
        # Tiny-only exact reconstruction including I8 representation/scales.
        for name,param in root.named_parameters():
            torch.testing.assert_close(param.to(device).full_tensor(),state[name.removeprefix('network.')].to(device),atol=0,rtol=0)
        memories=[];max_errors=[]
        for step in range(5):
            with torch.inference_mode(False),torch.no_grad():
                tracker.begin(device);context=root('preprocess_text',(text.to(device),),{});tracker.finish(context)
                assert context.dtype==torch.float32
                assert_sharded(root,config.cpu_offload)
                tracker.begin(device)
                y=root('forward',([video.to(device),audio.to(device)],torch.tensor([700.],device=device),context),
                       {'transformer_options':{'sample_sigmas':torch.tensor([1.,.5,0.],device=device)},'minimax_payload':{'seed':17}})
                tracker.finish(y)
                assert_sharded(root,config.cpu_offload)
                for unit in units:unit.reshard()
            sync(device);assert_sharded(root,config.cpu_offload)
            if managed_pool is not None:managed_pool.assert_parameters(root)
            # CPU and CUDA half GEMMs need not be bitwise identical.
            torch.testing.assert_close(context.cpu(),ctx_ref,atol=1.,rtol=.01)
            for actual,expected in zip(y,y_ref):
                torch.testing.assert_close(actual.cpu(),expected,atol=.02,rtol=.01)
                max_errors.append((actual.cpu()-expected).abs().max().item())
            memories.append(memory(device))
        if memories[-1]['allocated']>memories[1]['allocated']+2**20:
            raise AssertionError('H3 persistent allocation grows after warmup')
        output['int8_convrot' if quant else 'fp16']={'status':'PASS','world_size':world,
            'sequence_mode':config.sequence_mode,'sequence_active':net.blocks[0].attn._ps_sequence['enabled'],
            'weight_placement':config.weight_placement,'ats':managed_pool.report() if managed_pool else None,
            'patch_fingerprint':net._ps_patch_fingerprint,'cpu_offload':config.cpu_offload,'pin_memory':config.pin_memory,
            'prefetch_blocks':config.prefetch_blocks,'shards':evidence,'memory':memories,'max_abs_output':max(max_errors),
            'forward_count':5,'mode':'eval+no_grad+inference_mode(False)','baseline':'tiny CPU mixed precision, not real checkpoint'}
        del root,net,reference,y,y_ref,state,context,ctx_ref,units
        gc.collect();torch.cuda.empty_cache()
    return output


def phase_cache_h3_probe(device, config, folder):
    """Real FSDP/NCCL cache round-trip on the tiny checkpoints from native_h3_probe.

    No full_tensor(), no production checkpoint. Uses the same backend/cache
    constructor as workers, including fresh ATS pool ownership on each resume.
    """
    from dataclasses import replace
    import weakref
    from .fsdp_backend import H3Backend, memory
    from .phase_cache import ResidentBackend
    from .patch_config import H3PatchConfig
    output = {}
    for quant in (False, True):
        checkpoint = Path(folder, f"tiny-h3-{quant}.safetensors")
        cfg = replace(config, precision="int8_fp16" if quant else "fp16")
        patch = H3PatchConfig(enabled=True)
        def factory(cached):
            pool = None
            if cfg.weight_placement == "ats":
                from .ats_memory import ManagedShardPool
                pool = ManagedShardPool(device)
            return H3Backend(checkpoint, cfg, device, patch, managed_pool=pool, local_state=cached)
        owner = ResidentBackend(factory(None), factory, device)
        old_graph = weakref.ref(owner._backend)
        text = torch.full((1,7,256),100000.,device=device)
        video = torch.zeros((1,24,2,8,8),device=device)
        audio = torch.zeros((1,32,2,5),device=device)
        context, _ = owner.call("preprocess_text", (text,), {})
        result, _ = owner.call("forward", ([video,audio],torch.tensor([700.],device=device),context), {})
        reference = [t.cpu() for t in result]
        del result, context
        owner.end_run()
        parked = owner.idle()
        if not cfg.cpu_offload:
            if old_graph() is not None or owner._backend is not None:
                raise AssertionError("Parking retained a live FSDP graph")
            if any(t.device.type != "cpu" for t in owner._cached.parameters.values()):
                raise AssertionError("Parking retained non-CPU weights")
        context, resume = owner.call("preprocess_text", (text,), {})
        result, metrics = owner.call("forward", ([video,audio],torch.tensor([700.],device=device),context), {})
        for tensor, expected in zip(result, reference):
            torch.testing.assert_close(tensor.cpu(), expected, rtol=.003, atol=.002)
        if not cfg.cpu_offload:
            assert resume["phase_cache"]["resumed_from_ram"]
            assert resume["phase_cache"]["checkpoint_weight_reads_on_resume"] == 0
        owner.end_run()
        parked_again = owner.idle()
        output["int8_convrot" if quant else "fp16"] = dict(status="PASS",world_size=dist.get_world_size(),
            weight_placement=cfg.weight_placement,phase_offload=parked,phase_offload_again=parked_again,
            resume=resume["phase_cache"],memory=memory(device),forward=metrics,
            note="Tiny weights only; allocated is logical, ATS physical residency needs NVML/driver sampling")
        del owner, text, video, audio, result, context, reference
        gc.collect()
        torch.cuda.empty_cache()
    return output
