"""Session policy/probes вне CUDA context ComfyUI и согласованный worker dispatch."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings


def policy_fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def isolated_probe(provider,device,head_dim=128,heads=4,timeout=60,kv_heads=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),env.get("PYTHONPATH","")])
    argv=[sys.executable,"-m","powershard.attention_probe",provider,str(head_dim),str(heads)]
    if kv_heads:
        argv.append(str(kv_heads))
    try:
        run = subprocess.run(argv,env=env,capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(status="FAIL",reason="isolated CUDA probe timeout",uuid=device["uuid"])
    if run.returncode:
        return dict(status="FAIL",reason=f"isolated probe exit {run.returncode}: {run.stderr[-1800:]}",uuid=device["uuid"])
    try:result = json.loads(run.stdout)
    except json.JSONDecodeError:return dict(status="FAIL",reason="Некорректный probe protocol",stdout=run.stdout[-1000:])
    result["uuid"] = device["uuid"]
    return result


def prepare_policy(config,devices,checkpoint=None,geometry=None):
    requested = config.requested_attention
    head_dim,heads = 128,4
    kv_heads = None
    if checkpoint:
        from .checkpoint import Checkpoint
        shape = Checkpoint(checkpoint).model_config()
        head_dim,heads = shape["attention_head_dim"],shape["num_attention_heads"]
    if geometry is not None:
        head_dim,heads=geometry["head_dim"],geometry["heads"]
        # GQA-геометрия (Qwen 64Q/8KV): probe сертифицирует GQA-путь сборки.
        kv_heads = geometry.get("kv_heads")
    candidates = ["vllm_flash_attn","flash_attn","sdpa","math"] if requested=="auto" else [requested]
    if config.allow_fallback:
        if requested == "flash_attn":candidates.append("vllm_flash_attn")
        candidates += ["sdpa","math"]
    reports = {}
    probe_heads = [heads]
    if checkpoint and config.sequence_mode == "ulysses":
        # Token refiner uses the complete head set, sequence blocks use
        # ceil(H / world) heads. Custom Volta kernels must pass BOTH shapes.
        probe_heads = list(dict.fromkeys([heads, (heads+len(devices)-1)//len(devices)]))
    for provider in dict.fromkeys(candidates):
        reports[provider] = [dict(isolated_probe(provider,d,head_dim,h,90,kv_heads),probe_heads=h)
                             for d in devices for h in probe_heads]
        if all(r["status"]=="PASS" for r in reports[provider]):
            policy = dict(requested=requested,effective=provider,probes=reports,
                          geometry=dict(head_dim=head_dim,heads=heads,dtype="float16",layout="BLHD",
                                        kv_heads=kv_heads or heads,certified_heads=probe_heads,
                                        gqa_certified=bool(kv_heads and kv_heads!=heads)),
                          devices=[d["uuid"] for d in devices],allow_fallback=config.allow_fallback,
                          reason=None if requested in (provider,"auto") else reports.get(requested),
                          provider_identities=[r["identity"] for r in reports[provider]])
            policy["fingerprint"] = policy_fingerprint(policy)
            if policy["reason"]:
                reason_text = json.dumps(policy["reason"], ensure_ascii=False)
                if requested == "flash_attn" and "flash_attn_2_cuda" in reason_text:
                    reason_text += (" — нужен установленный custom provider для ppc64le/sm_70; "
                                    "auto проверит оба поддерживаемых интерфейса вашей сборки")
                warnings.warn(f"Requested: {requested}; Effective: {provider}; Reason: {reason_text}")
            return policy
    raise RuntimeError("Нет проверенного attention пути на всех выбранных GPU: "+json.dumps(reports,ensure_ascii=False))


class AttentionDispatcher:
    def __init__(self,config,policy=None):
        self.config = config
        self.requested = config.requested_attention
        self.policy = policy or dict(requested=self.requested,effective="math" if self.requested=="auto" else self.requested,
                                    reason="Локальный вызов без CUDA probe; не является CUDA сертификацией")
        self.effective = self.policy["effective"]
        self.counts,self.reasons = Counter(),{}
        self.memory_estimate = {}
        self._memory_warned = False
        self.provider = None
        if self.effective in ("flash_attn","vllm_flash_attn"):
            from .attention_providers import FlashProvider
            from .vllm_adapter import VLLMFlashAdapter
            self.provider = VLLMFlashAdapter() if self.effective=="vllm_flash_attn" else FlashProvider()
        elif self.effective == "sageattention":
            from .attention_providers import SageProvider
            self.provider = SageProvider()
        if self.provider and self.policy.get("provider_identities"):
            if self.provider.identity not in self.policy["provider_identities"]:
                raise RuntimeError("Attention provider изменился между probe и worker; перезапустите session")

    def __call__(self,q,k,v,options=None,group="unspecified",compute_fp16=False):
        from .attention_contract import AttentionOptions,UnsupportedAttention,validate,math_attention,sdpa_attention
        o = options or AttentionOptions()
        validate(q,k,v,o)
        effective = self.effective
        b,lq,h,d=q.shape;lk=k.shape[1]
        self.memory_estimate = dict(dense_score_bytes=b*h*lq*lk*4,
            math_score_tile_bytes=b*h*min(lq,self.config.query_chunk)*min(lk,self.config.key_chunk)*4,
            output_bytes=b*lq*h*v.shape[-1]*q.element_size(),
            note="Оценки, не measured peak. SDPA kernel/workspace зависит от automatic dispatch; GQA replication/dequant/activations дополнительно.")
        if self.memory_estimate['dense_score_bytes']>2**30 and not self._memory_warned:
            warnings.warn(f"Attention dense score estimate {self.memory_estimate['dense_score_bytes']/2**30:.2f} GiB; math использует tiles, SDPA peak зависит от dispatch. Попытка не запрещена.")
            self._memory_warned = True
        try:
            geometry = self.policy.get("geometry")
            if geometry and effective not in ("math","sdpa"):
                # CUDA probe подтверждает только эту область. Более сложный
                # контракт не объявляется рабочим по одной лишь signature.
                if q.shape[-1]!=geometry["head_dim"] or str(q.dtype)!="torch."+geometry["dtype"]:
                    raise UnsupportedAttention("dtype/head_dim вне области CUDA probe этой session")
                if q.shape[2] not in geometry.get("certified_heads", [geometry["heads"]]):
                    raise UnsupportedAttention("Число heads вне области CUDA probe этой session")
                gqa_certified = geometry.get("gqa_certified", False)
                is_gqa = q.shape[2] != k.shape[2]
                if is_gqa and not gqa_certified:
                    raise UnsupportedAttention("GQA не сертифицирован probe этой сборки; используем семантически точный путь")
                if not is_gqa and o.causal and q.shape[1]!=k.shape[1]:
                    raise UnsupportedAttention("rectangular causal не выполнен probe; используем семантически точный путь")
                if o.dropout_p or o.window_size!=(-1,-1) or o.alibi_slopes is not None:
                    raise UnsupportedAttention("Для этой custom CUDA сборки не выполнен probe dropout/window/ALiBi; используем семантически точный путь")
            if effective=="math":
                result = math_attention(q,k,v,o,self.config.query_chunk,self.config.key_chunk,compute_fp16)
            elif effective=="sdpa":result = sdpa_attention(q,k,v,o)
            else:result = self.provider(q,k,v,o)
        except UnsupportedAttention as error:
            # ТОЛЬКО проверяемый контракт до kernel; RuntimeError/CUDA/OOM идут наверх.
            if not self.config.allow_fallback:raise
            reason = str(error)
            first = reason not in self.reasons
            effective = "sdpa"
            try:result = sdpa_attention(q,k,v,o)
            except UnsupportedAttention:
                effective = "math"
                result = math_attention(q,k,v,o,self.config.query_chunk,self.config.key_chunk,compute_fp16)
            self.reasons[reason] = effective
            if first:
                warnings.warn(f"Requested: {self.requested}; Effective: {effective}; Reason: {reason}")
        self.counts[group+":"+effective] += 1
        return result

    def report(self):
        used={}
        for label, count in self.counts.items():
            group, provider = label.rsplit(":",1)
            used.setdefault(group,{})[provider]=count
        return dict(requested=self.requested,effective=self.effective,
                    effective_used=used,
                    effective_note="selected policy; effective_used/call_counts are actual provider calls by group, not CUDA kernel identity",
                    dispatch="SDPA automatic dispatch" if self.effective=="sdpa" else self.effective,
                    provider=self.provider.identity if self.provider else {"module":"torch"},
                    policy=self.policy,call_counts=dict(self.counts),semantic_fallbacks=self.reasons.copy(),memory_estimate=self.memory_estimate.copy())
