"""Worker-local target-feature forecasting; gates стоят СНАРУЖИ FSDP __call__.

Формула Chebyshev/ridge + linear extrapolation реализована здесь самостоятельно.
Spectrum: hanjq17/Spectrum (MIT); H3 target-only idea: xmarre/ComfyUI-Spectrum-MiniMax-H3
(GPL-3.0-or-later). Исходные веса, RoPE и native final audio/video heads не меняются.
Это приближённое ускорение, отдельный opt-in, не exact attention.
"""
from collections import OrderedDict, Counter
import math
import time
import torch
from torch import nn
import torch.distributed as dist
from .spectrum_config import SpectrumConfig
from .config import shard_bounds


def forecast_weights(points, target, degree, ridge, blend):
    """Маленькая CPU float64 система; большие features в solve не участвуют."""
    if len(points) < degree+1 or len(set(points)) != len(points):
        raise ValueError("Spectrum: недостаточные/повторные координаты history")
    def basis(x):
        values=[torch.ones_like(x),x]
        for _ in range(2,degree+1):values.append(2*x*values[-1]-values[-2])
        return torch.stack(values[:degree+1],dim=-1)
    x=torch.tensor(points,dtype=torch.float64)
    phi=basis(x);at=basis(torch.tensor(target,dtype=torch.float64))
    spectral=at @ torch.linalg.solve(phi.T@phi+ridge*torch.eye(degree+1,dtype=torch.float64),phi.T)
    linear=torch.zeros_like(spectral)
    ratio=(target-points[-1])/(points[-1]-points[-2])
    linear[-2]=-ratio;linear[-1]=1+ratio
    result=blend*spectral+(1-blend)*linear
    if not bool(torch.isfinite(result).all()):raise FloatingPointError("Spectrum coefficient solve non-finite")
    return result.tolist()


class SpectrumEngine:
    def __init__(self, config=None, device="cpu", rank=None, world=None):
        self.config=config or SpectrumConfig()
        self.device=torch.device(device)
        self.world=world if world is not None else (dist.get_world_size() if dist.is_initialized() else 1)
        self.rank=rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.entries=OrderedDict();self.bytes=0;self.run_id=None;self.metadata={}
        self.forecast=False;self.entry=None;self.coordinate=None;self.previous_report=None
        self.stats=Counter();self.reasons=Counter()

    def clear(self):
        self.previous_report=self.report()
        self.entries.clear();self.bytes=0;self.run_id=None;self.entry=None;self.forecast=False

    def begin(self, metadata):
        self.metadata=metadata or {};self.forecast=False;self.entry=None
        run=self.metadata.get("run_id")
        if run != self.run_id:
            self.clear();self.run_id=run;self.stats.clear();self.reasons.clear()
        reason=self.metadata.get("reason","missing_native_sampler_context")
        eligible=self.config.enabled and self.metadata.get("eligible",False)
        if not self.config.enabled:reason="disabled"
        elif eligible:
            key=self.metadata["conditioning_key"]
            self.entry=self.entries.get(key)
            if self.entry is not None:self.entries.move_to_end(key)
            index=self.metadata["step_index"];self.coordinate=self.metadata["coordinate"]
            if self.entry and index <= self.entry["last_index"]:
                self.bytes-=self.entry["bytes"];del self.entries[key];self.entry=None
                reason="repeated_or_nonmonotonic_step"
            elif index < self.config.warmup:reason="warmup"
            elif index >= self.metadata["steps"]-self.config.tail:reason="tail"
            elif not self.entry or len(self.entry["history"]) < self.config.degree+1:reason="history_not_ready_or_budget"
            elif self.entry["forecast_streak"] >= self.config.max_forecast:reason="actual_refresh"
            else:self.forecast=True;reason="forecast"
        # ВСЕ ranks согласуют решение до root/FSDP forward, также при локальном veto.
        if self.config.enabled and dist.is_initialized():
            vote=torch.tensor(int(self.forecast),device=self.device,dtype=torch.int32)
            dist.all_reduce(vote,op=dist.ReduceOp.MIN)
            dist.broadcast(vote,src=0)
            agreed=bool(vote.cpu())  # ровно одна граница RPC, не каждый DiT block
            if self.forecast and not agreed:reason="rank_veto"
            self.forecast=agreed
        self.reason=reason;self.stats["forecast" if self.forecast else "actual"]+=1
        self.reasons[reason]+=1

    @staticmethod
    def target_slices(layout):
        selected=[(a,b,kind) for a,b,kind in layout.segments if kind in ("audio","video")]
        if len(selected)!=2 or {s[2] for s in selected}!={"audio","video"}:
            raise ValueError("Spectrum: native H3 target audio/video segments отсутствуют")
        return tuple(sorted(selected))

    def _local_target(self,h,segments):
        total=sum(b-a for a,b,_ in segments);start,end=shard_bounds(total,self.rank,self.world)
        out=h.new_empty((end-start,h.shape[-1]),dtype=torch.float32)
        offset=0
        for a,b,_ in segments:
            lo,hi=max(start,offset),min(end,offset+b-a)
            if hi>lo:out[lo-start:hi-start].copy_(h[a+lo-offset:a+hi-offset].float())
            offset+=b-a
        return out,total

    def capture(self,h,layout):
        if not self.config.enabled or not self.metadata.get("eligible",False):return
        segments=self.target_slices(layout)
        total=sum(b-a for a,b,_ in segments);a,b=shard_bounds(total,self.rank,self.world)
        size=(b-a)*h.shape[-1]*4;limit=self.config.history_mib*2**20
        # Не хранить history, которая заведомо не сможет содержать degree+1 anchors.
        if not limit or size*(self.config.degree+1)>limit:
            self.reasons["history_budget_too_small"]+=1;return
        local,total=self._local_target(h,segments)
        key=self.metadata["conditioning_key"]
        entry=self.entries.get(key)
        signature=(tuple(h.shape),segments)
        if entry is not None and entry["signature"]!=signature:
            self.bytes-=entry["bytes"];del self.entries[key];entry=None
        if entry is None:
            if len(self.entries)>=256:
                _,old=self.entries.popitem(last=False);self.bytes-=old["bytes"]
            entry=dict(history=[],bytes=0,signature=signature,total=total,forecast_streak=0,last_index=-1)
            self.entries[key]=entry
        while entry["history"] and (len(entry["history"])>=self.config.history_size or entry["bytes"]+size>limit):
            _,old=entry["history"].pop(0);n=old.numel()*old.element_size();entry["bytes"]-=n;self.bytes-=n
        while self.bytes+size>limit:
            other=next((k for k in self.entries if k!=key),None)
            if other is None:
                _,old=entry["history"].pop(0);n=old.numel()*old.element_size();entry["bytes"]-=n;self.bytes-=n
            else:self.bytes-=self.entries.pop(other)["bytes"]
        target="cpu" if self.config.history_device=="cpu" else self.device
        saved=local.to(target,copy=True) # owned snapshot; не alias reused forward tensor
        entry["history"].append((self.coordinate,saved));entry["bytes"]+=size;self.bytes+=size
        entry["last_index"]=self.metadata["step_index"];entry["forecast_streak"]=0
        self.entry=entry;self.stats["captured_local_bytes"]+=size
        self.stats["peak_history_bytes"]=max(self.stats["peak_history_bytes"],self.bytes)

    def predict(self,h,layout):
        start=time.perf_counter();entry=self.entry;segments=self.target_slices(layout)
        if entry is None or entry["signature"]!=(tuple(h.shape),segments):
            # После consensus не меняем ветку локально и не повторяем повреждённый forward.
            raise RuntimeError("Spectrum: layout изменился после решения ranks; session должна быть завершена")
        points=[x for x,_ in entry["history"]]
        weights={kind:forecast_weights(points,self.coordinate,self.config.degree,self.config.ridge,
            self.config.audio_blend if kind=="audio" else self.config.blend) for kind in ("audio","video")}
        template=entry["history"][-1][1];local=torch.zeros(template.shape,device=h.device,dtype=torch.float32)
        # CPU history загружается кусками, без полной GPU копии всех anchors.
        rows=max(1,(16*2**20)//max(4*h.shape[-1],1))
        lo,hi=shard_bounds(entry["total"],self.rank,self.world);offset=0
        for a,b,kind in segments:
            first,last=max(lo,offset)-lo,min(hi,offset+b-a)-lo
            for begin in range(max(0,first),max(0,last),rows):
                end=min(begin+rows,last)
                for weight,(_,feature) in zip(weights[kind],entry["history"]):
                    local[begin:end].add_(feature[begin:end].to(h.device),alpha=weight)
            offset+=b-a
        if dist.is_initialized() and self.world>1:
            from .attention import gather_rows
            compact=gather_rows(local,entry["total"])
            self.stats["forecast_target_all_gathers"]+=1
        else:compact=local
        if compact.shape[0]!=entry["total"]:raise RuntimeError("Spectrum target shard reconstruction failed")
        offset=0
        for a,b,_ in segments:h[a:b].copy_(compact[offset:offset+b-a]);offset+=b-a
        entry["last_index"]=self.metadata["step_index"];entry["forecast_streak"]+=1
        self.stats["forecast_host_enqueue_s"]+=time.perf_counter()-start
        return h

    def report(self):
        return dict(enabled=self.config.enabled,mode="target_feature_forecast",run_id=self.run_id,
            decision="FORECAST" if self.forecast else "ACTUAL",reason=getattr(self,"reason","not_called"),
            counters=dict(self.stats),reasons=dict(self.reasons),history_bytes=self.bytes,
            history_limit_bytes=self.config.history_mib*2**20,history_device=self.config.history_device,
            history_scope="local target rows, all conditioning branches share this rank budget",
            approximate=True,final_heads="native_every_forward",config=self.config.to_dict())


class SpectrumBlockGate(nn.Module):
    def __init__(self, block, engine, index, count):
        super().__init__();self.block=block;self.engine=engine;self.index=index;self.count=count

    @property
    def attn(self):return self.block.attn

    def set_modules_to_forward_prefetch(self,modules):
        return self.block.set_modules_to_forward_prefetch([m.block if isinstance(m,SpectrumBlockGate) else m for m in modules])

    def forward(self,h,*args,**kwargs):
        layout=kwargs["transformer_options"]["minimax_h3_layout"]
        if self.engine.forecast:
            return self.engine.predict(h,layout) if self.index==0 else h
        result=self.block(h,*args,**kwargs) # FSDP __call__ и hooks только на ACTUAL
        if self.index==self.count-1:self.engine.capture(result,layout)
        return result


def install_spectrum(net,engine):
    if any(isinstance(b,SpectrumBlockGate) for b in net.blocks):
        raise ValueError("Spectrum gates уже установлены на этом worker module")
    if engine.config.enabled:
        net.blocks=nn.ModuleList([SpectrumBlockGate(b,engine,i,len(net.blocks)) for i,b in enumerate(net.blocks)])
