"""Capability detection. Версия/SHA ComfyUI не являются условием допуска."""
import importlib
import inspect


def require_signature(function, arguments, label):
    if not callable(function):
        raise RuntimeError(f"PowerShard: отсутствует callable {label}")
    signature = inspect.signature(function)
    variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    missing = set(arguments)-set(signature.parameters)
    if missing and not variadic:
        raise RuntimeError(f"PowerShard API {label}{signature}: отсутствуют arguments {sorted(missing)}")
    return str(signature)


def verify_comfy(path):
    report = {"admission": "capabilities_only", "signatures": {}}
    required = {
        "comfy.ldm.minimax.model": {"MiniMaxH3Model": {"forward": ("x", "timestep", "context"),
            "preprocess_text_embeds": ("text_states",)}, "DiTBlock": {"forward": ("x", "t_emb")},
            "MLP": {"forward": ("x",)}},
        "comfy.model_patcher": {"ModelPatcher": {"clone": (), "load": (), "partially_unload": (),
            "add_wrapper_with_key": (), "cleanup": ()}},
        "comfy.model_base": {"MiniMaxH3": {"extra_conds": (), "get_dtype_inference": ()}},
    }
    for module_name, classes in required.items():
        module = importlib.import_module(module_name)
        for class_name, methods in classes.items():
            cls = getattr(module, class_name, None)
            for method, args in methods.items():
                label = f"{module_name}.{class_name}.{method}"
                report["signatures"][label] = require_signature(getattr(cls, method, None), args, label)
    for name in ("comfy.samplers", "comfy.patcher_extension", "comfy_api"):
        module = importlib.import_module(name)
        report[name] = getattr(module, "__file__", "namespace")
    samplers = importlib.import_module("comfy.samplers")
    helpers = importlib.import_module("comfy.sampler_helpers")
    report["signatures"]["CFGGuider.sample"] = require_signature(samplers.CFGGuider.sample,
        ("noise", "latent_image", "sampler", "sigmas"), "CFGGuider.sample")
    report["signatures"]["prepare_sampling"] = require_signature(helpers.prepare_sampling,
        ("model", "noise_shape", "conds"), "prepare_sampling")
    return report


def check_h3_instance(net):
    for name in ("condition_proj", "blocks", "token_refiner", "final_layer", "preprocess_text_embeds"):
        if not hasattr(net, name):
            raise RuntimeError(f"H3 capability: отсутствует {name}")
    for block in list(net.blocks)+list(net.token_refiner.blocks):
        for name in ("norm1", "norm2", "attn", "mlp"):
            if not hasattr(block, name):
                raise RuntimeError(f"H3 block capability: отсутствует {name}")
        for name in ("qkv_proj", "out_proj", "q_norm", "k_norm", "heads", "head_dim"):
            if not hasattr(block.attn, name):
                raise RuntimeError(f"H3 attention capability: отсутствует {name}")
        if not all(hasattr(block.mlp, name) for name in ("fc1", "fc2")):
            raise RuntimeError("H3 MLP: нужен SwiGLU fc1/fc2 interface")
    return True
