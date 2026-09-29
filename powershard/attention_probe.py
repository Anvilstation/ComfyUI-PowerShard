"""Одноразовый CUDA probe. Никогда не импортируется для запуска kernel из nodes."""
import json
import sys


def probe(provider,head_dim=128,heads=4,device="cuda:0",kv_heads=None):
    import torch
    from .attention_contract import AttentionOptions,math_attention,sdpa_attention
    from .attention_providers import FlashProvider,SageProvider,module_identity
    if provider in ("flash_attn","vllm_flash_attn"):
        adapter=FlashProvider(provider);fn=adapter;identity=adapter.identity
    elif provider=="sageattention":
        adapter=SageProvider();fn=adapter;identity=adapter.identity
    else:
        fn=sdpa_attention if provider=="sdpa" else math_attention
        identity=module_identity("torch",torch)
    # GQA-сертификация: kv_heads передаётся для Qwen-геометрии (64Q/8KV).
    # Когда задана — часть кейсов гоняется с GQA (hq % hk == 0, hq != hk),
    # чтобы policy-гейт мог разрешить GQA-контракт этой CUDA-сборке.
    gqa = None
    if kv_heads and kv_heads != heads and heads % kv_heads == 0:
        gqa = dict(q_heads=heads, kv_heads=kv_heads)
    generator=torch.Generator(device=device).manual_seed(1729)
    errors=[]
    # B>1, cross lengths, non-default scale, non-contiguous input, ordinary/causal.
    # С GQA: короткие cross-кейсы на GQA-геометрии + один causal equal-heads
    # (causal GQA varlen у custom-сборок часто отдельный кодовый путь).
    cases = [(5,7,.13,False),(5,7,.37,False),(7,7,.2,True)]
    for lq,lk,scale,causal in cases:
        if gqa and not causal:
            qh, kh = gqa["q_heads"], gqa["kv_heads"]
        else:
            qh = kh = heads
        q=torch.randn((2,lq,qh,head_dim*2),device=device,dtype=torch.float16,generator=generator)[...,::2]
        k=torch.randn((2,lk,kh,head_dim),device=device,dtype=torch.float16,generator=generator)
        v=torch.randn((2,lk,kh,head_dim),device=device,dtype=torch.float16,generator=generator)
        o=AttentionOptions(softmax_scale=scale,causal=causal)
        reference=math_attention(q.float(),k.float(),v.float(),o)
        for _ in range(2):actual=fn(q,k,v,o)
        torch.cuda.synchronize(device) if torch.device(device).type=="cuda" else None
        atol,rtol=(.08,.08) if provider=="sageattention" else (.007,.007)
        if not torch.isfinite(actual).all():raise FloatingPointError("probe produced NaN/Inf")
        torch.testing.assert_close(actual.float(),reference,atol=atol,rtol=rtol)
        errors.append(dict(lq=lq,lk=lk,scale=scale,causal=causal,gqa=qh!=kh,max_abs=(actual.float()-reference).abs().max().item()))
    return dict(status="PASS",import_result="PASS",kernel_result="PASS",numerical_result="PASS",
                identity=identity,geometry=dict(head_dim=head_dim,heads=heads,dtype="float16",layout="BLHD",
                kv_heads=kv_heads or heads,gqa_certified=bool(gqa)),
                extensions={name:getattr(module,"__file__",None) for name,module in list(sys.modules.items())
                            if ("flash" in name or "sage" in name) and str(getattr(module,"__file__","")).endswith('.so')},
                torch=str(torch.__version__),cuda_build=torch.version.cuda,
                device=torch.cuda.get_device_name(device) if torch.device(device).type=="cuda" else str(device),
                checks=errors,tolerance=dict(atol=atol,rtol=rtol),
                note="Небольшой probe не доказывает качество H3/длинных sequences или другие dtypes/head_dim")


if __name__=="__main__":
    # Здесь можно ловить все ошибки: процесс завершается, испорченный context не переиспользуется.
    try:result=probe(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),
                     kv_heads=int(sys.argv[4]) if len(sys.argv)>4 else None)
    except Exception as error:result=dict(status="FAIL",reason=f"{type(error).__name__}: {error}")
    print(json.dumps(result))
