#!/usr/bin/env python3
"""Real load-only or load+forward memory audit; worker stats, not host CUDA stats.

python tests/audit_memory.py MODEL --comfy /opt/ComfyUI --sets 3 4 5 6 --placement cpu
Add --forward for a tiny synthetic native H3 audio/video forward.
ATS allocator byte counts are logical, not physical GPU residency.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def nvidia_smi_rows():
    try:
        run = subprocess.run(['nvidia-smi', '--query-gpu=uuid,index,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        return {parts[0]: dict(physical_index=int(parts[1]), used_mib=int(parts[2]), total_mib=int(parts[3]))
                for line in run.stdout.splitlines() if line.strip()
                for parts in [[p.strip() for p in line.split(',')]]}
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        return dict(error=str(error))


def worker_memory(session):
    from powershard.topology import process_memory
    workers = []
    for rank, proc in enumerate(session.processes):
        try:
            status = Path(f'/proc/{proc.pid}/status').read_text()
            fields = {l.split(':',1)[0]: l.split(':',1)[1].strip() for l in status.splitlines() if ':' in l}
            row = dict(rank=rank, pid=proc.pid,
                       rss_bytes=int(fields.get('VmRSS','0 kB').split()[0])*1024,
                       locked_bytes=int(fields.get('VmLck','0 kB').split()[0])*1024)
        except OSError as error:
            row = dict(rank=rank, pid=proc.pid, error=str(error))
        workers.append(row)
    return dict(host=process_memory(), workers=workers, driver_by_uuid=nvidia_smi_rows())


def audit_one(size, checkpoint, args, visible):
    import torch
    from powershard.config import DistributedConfig
    from powershard.comfy_adapter import load_model
    from powershard.checkpoint import Checkpoint
    selected = visible[:size]
    if len(selected) != size: return dict(status='NOT_RUN', requested_world=size, reason='Not enough CUDA-visible devices')
    folder = Path(args.output)/f'{args.placement}-{args.sequence_mode}-{size}gpu'
    folder.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(checkpoint)
    cfg = DistributedConfig(gpu_ids=tuple(d['user_id'] for d in selected), weight_placement=args.placement,
            precision='int8_fp16' if ck.quantization() else 'fp16', sequence_mode=args.sequence_mode,
            attention_backend=args.attention_backend, reserve_gib=args.reserve_gib,
            prefetch_blocks=args.prefetch_blocks, pin_memory=args.pin_memory,
            numa_policy=args.numa_policy, release_after_sampling=False)
    result = dict(status='STARTED', world_size=size, selected_devices=selected, config=cfg.to_dict(),
                  checkpoint=ck.identity(), checkpoint_file_bytes=ck.path.stat().st_size,
                  allocator_counts_note='ATS bytes are logical; driver memory is physical, shared with other processes',
                  baseline_driver_by_uuid=nvidia_smi_rows())
    start = time.perf_counter()
    patcher = load_model(str(ck.path), cfg, report_dir=folder)
    result['host_meta_setup_s'] = time.perf_counter()-start
    session = patcher.session
    try:
        start = time.perf_counter()
        session.start()  # Explicitly materialize shards BEFORE the load-only snapshot.
        result['worker_start_and_load_s'] = time.perf_counter()-start
        result['after_load'] = worker_memory(session)
        result['rank_preflight'] = session.history[-1]['ranks']
        result['forward'] = dict(status='NOT_RUN', reason='--forward not supplied')
        if args.forward:
            shape = ck.model_config()
            gen = torch.Generator().manual_seed(44)
            video = torch.randn(1,shape['latents_dim'],2,8,8,generator=gen)
            audio = torch.randn(1,shape['audio_latents_dim'],2,5,generator=gen)
            text = torch.randn(1,7,shape['text_dim'],generator=gen)
            start = time.perf_counter()
            context = session.call('preprocess_text',(text,),{})
            outputs = session.call('forward',([video,audio],torch.tensor([700.]),context),
                            {'transformer_options':{'sample_sigmas':torch.tensor([1.,.5,0.])}})
            assert len(outputs)==2
            for got, expected in zip(outputs,(video,audio)):
                assert got.shape==expected.shape and torch.isfinite(got).all()
            result['forward'] = dict(status='PASS', scope='tiny synthetic forward, not generation quality',
                                     wall_s=time.perf_counter()-start, rank_metrics=session.history[-1]['ranks'],
                                     after_forward=worker_memory(session))
        result['status'] = 'PASS'
    except Exception as error:
        result['status']='FAIL'; result['error']=str(error)
        result['last_worker_memory']=worker_memory(session)
    finally:
        session.close()
        result['after_close_driver_by_uuid']=nvidia_smi_rows()
        (folder/'memory-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('--comfy','--comfy-path',dest='comfy')
    p.add_argument('--sets',type=int,nargs='+',default=[3,4,5,6]); p.add_argument('--gpus',default='all')
    p.add_argument('--placement','--config',dest='placement',choices=['gpu','cpu','ats'],default='gpu')
    p.add_argument('--sequence-mode',choices=['token','ulysses'],default='ulysses')
    p.add_argument('--forward',action='store_true'); p.add_argument('--attention-backend',default='auto')
    p.add_argument('--reserve-gib',type=float,default=2.); p.add_argument('--prefetch-blocks',type=int,choices=[0,1,2],default=0)
    p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--numa-policy',choices=['none','auto','bind'],default='auto')
    p.add_argument('--output',default='reports/local-memory-audit'); args=p.parse_args()
    if any(n<1 for n in args.sets): p.error('--sets must be positive')
    import torch
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():
        report=dict(status='NOT_RUN',reason='CUDA unavailable',torch=torch.__version__)
        (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return 2
    candidates=[Path(args.comfy)] if args.comfy else list(Path(__file__).resolve().parents)
    comfy=next((path.resolve() for path in candidates if (path/'comfy/model_base.py').is_file()),None)
    if comfy is None:p.error('Specify --comfy /opt/ComfyUI')
    sys.path.insert(0,str(comfy));sys.argv=['memory-audit','--disable-dynamic-vram','--use-pytorch-cross-attention']
    import comfy.options
    comfy.options.enable_args_parsing()
    from powershard.devices import resolve_gpu_selection
    visible=resolve_gpu_selection(tuple(args.gpus.split(',')))
    results=[]
    for size in args.sets:
        result=audit_one(size,args.checkpoint,args,visible);results.append(result)
        print(json.dumps(dict(world_size=size,status=result['status'],placement=args.placement,error=result.get('error')),ensure_ascii=False),flush=True)
    (folder/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    return 0 if all(r['status']=='PASS' for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
