"""Прерывание шага в workers по Cancel (Wan / LTX / MiniMax H3 quant).

Раньше Cancel ждал конца текущего шага: на длинных шагах (V100, медленный бэкенд attention) GPU продолжали
считать минутами, VRAM не освобождалась. Теперь host при Cancel пишет `<session_dir>/cancel` с номером
прерываемой команды, а workers перед каждым DiT-блоком проверяют файл и согласуют решение all-reduce MAX —
все rank выходят на одном и том же блоке (иначе разошлись бы коллективы FSDP/sequence parallel).
Шаг заканчивается исключением StepCancelled; worker отвечает host `cancelled` и продолжает работу
(дальше дренаж делает end_run + парковку, как раньше).
"""
import os

_STATE = {"path": None, "sequence": None, "device": None, "every": 1, "count": 0}


class StepCancelled(RuntimeError):
    pass


def configure(session_dir, device):
    _STATE["path"] = os.path.join(session_dir, "cancel")
    _STATE["device"] = device
    try:
        _STATE["every"] = max(1, int(os.environ.get("POWERSHARD_CANCEL_CHECK_BLOCKS", "1")))
    except ValueError:
        _STATE["every"] = 1


def begin(sequence):
    _STATE["sequence"] = str(sequence)
    _STATE["count"] = 0


def end():
    _STATE["sequence"] = None


def requested_locally():
    path, sequence = _STATE["path"], _STATE["sequence"]
    if path is None or sequence is None or not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return f.read().strip() == sequence
    except OSError:
        return False


def check():
    """Вызывается перед каждым блоком на ВСЕХ rank (одинаковое число раз) — согласованное решение."""
    if _STATE["sequence"] is None or _STATE["path"] is None:
        return
    _STATE["count"] += 1
    if _STATE["count"] % _STATE["every"]:
        return
    import torch
    import torch.distributed as dist
    flag = 1.0 if requested_locally() else 0.0
    if dist.is_initialized() and dist.get_world_size() > 1:
        value = torch.tensor([flag], dtype=torch.float32, device=_STATE["device"])
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        flag = float(value.item())
    if flag > 0:
        raise StepCancelled("шаг прерван по Cancel")


def install_block_hooks(blocks):
    """forward_pre_hook (prepend: раньше unshard FSDP) на каждый DiT-блок."""
    def hook(module, args, kwargs=None):
        check()
    for block in blocks:
        try:
            block.register_forward_pre_hook(hook, prepend=True)
        except TypeError:  # старый torch без prepend
            block.register_forward_pre_hook(hook)
