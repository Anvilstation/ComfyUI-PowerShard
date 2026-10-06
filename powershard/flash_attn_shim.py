"""Scoped Flash dense API over the user's varlen-only vLLM build.

Never registered globally as ``flash_attn`` and never modifies sys.path.
Independent Q/K lengths and all requested options reach the real provider.
"""
import vllm_flash_attn as implementation
from .attention_contract import AttentionOptions, UnsupportedAttention

__version__="0.5.2+scoped-vllm-shim"
# The actual entrypoint is preferred by FlashProvider. No dense re-packing
# through a wrapper that assumes Lq == Lk or drops the caller's scale.
flash_attn_varlen_func=implementation.flash_attn_varlen_func


def flash_attn_func(q,k,v,dropout_p=0.,softmax_scale=None,causal=False,
                    window_size=(-1,-1),alibi_slopes=None,deterministic=False,
                    return_attn_probs=False,**kwargs):
    if kwargs:raise UnsupportedAttention("Unrecognized Flash shim options: "+", ".join(sorted(kwargs)))
    from .vllm_adapter import VLLMFlashAdapter
    options=AttentionOptions(dropout_p=dropout_p,softmax_scale=softmax_scale,
        causal=causal,causal_alignment="bottom_right",window_size=window_size,
        alibi_slopes=alibi_slopes,deterministic=deterministic,
        return_attn_probs=return_attn_probs)
    return VLLMFlashAdapter(implementation)(q,k,v,options)
