"""H3-Image: checkpoint-carried buffers must be mapped, not left as live CPU memory."""

import torch
from torch import nn

from powershard.image_backend import promote_file_buffers_to_meta


class _Rope(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(4, dtype=torch.float32))


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("adaln_t_table", torch.empty(1025, 8, dtype=torch.float32))
        self.rope = _Rope()


class _Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _Inner()
        self.register_buffer("pdd_sigmas", torch.tensor([1.0, 0.0]))


def test_buffers_present_in_checkpoint_move_to_meta():
    network = _Network()
    keys = {"inner.adaln_t_table", "diffusion_model.inner.rope.inv_freq"}
    identities = {}

    promote_file_buffers_to_meta(network, keys, identities)

    assert network.inner.adaln_t_table.device.type == "meta"
    assert network.inner.rope.inv_freq.device.type == "meta"
    # not in the checkpoint -> stays a real device buffer
    assert network.pdd_sigmas.device.type == "cpu"
    assert len(identities) == 2


def test_buffers_already_on_meta_are_skipped():
    network = _Network()
    network.inner.adaln_t_table = torch.empty(1025, 8, device="meta")
    identities = {}

    promote_file_buffers_to_meta(network, {"inner.adaln_t_table"}, identities)

    assert network.inner.adaln_t_table.device.type == "meta"
    assert identities == {}
