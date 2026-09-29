"""Единственный внутренний host wrapper. Не передаётся в CUDA subprocess."""
def guard_sampling(executor, model, noise_shape, conds, **kwargs):
    model.validate()
    for branch in conds.values():
        for cond in branch:
            for key in ("control", "hooks", "gligen", "additional_models"):
                if cond.get(key) is not None:
                    raise ValueError(f"PowerShard: неподдерживаемый {key} в conditioning; workers не запущены")
    return executor(model, noise_shape, conds, **kwargs)


def strip_internal_wrappers(value):
    # Сохраняем все сторонние объекты, чтобы обычная валидация отклонила их.
    out = {}
    for kind, keyed in value.items():
        out[kind] = {}
        for key, functions in keyed.items():
            allowed = kind == "prepare_sampling" and key == "powershard"
            from .spectrum_host import spectrum_outer_sample
            out[kind][key] = [f for f in functions if not ((allowed and f is guard_sampling) or
                (kind=="outer_sample" and key=="powershard_run" and f is spectrum_outer_sample))]
    return out
