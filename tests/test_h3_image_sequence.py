import types

import powershard.attention as attention
import powershard.image_backend as backend


class _Net:
    pass


class MiniMaxH3ImageModel(_Net):
    pass


class OtherModel(_Net):
    pass


def test_installs_wrappers_for_h3_image():
    calls = []
    attention.install_attention = lambda net, cfg: calls.append(("attn", net, cfg))
    attention.install_sequence = lambda net, cfg: calls.append(("seq", net, cfg))

    net = MiniMaxH3ImageModel()
    cfg = types.SimpleNamespace(sequence_mode="ulysses")
    families = backend.install_h3_image_sequence(net, cfg, [])

    assert families == ["minimax_h3_image"]
    assert [c[0] for c in calls] == ["attn", "seq"]
    assert all(c[1] is net and c[2] is cfg for c in calls)


def test_keeps_existing_families_untouched():
    calls = []
    attention.install_attention = lambda net, cfg: calls.append("attn")
    attention.install_sequence = lambda net, cfg: calls.append("seq")

    net = MiniMaxH3ImageModel()
    families = backend.install_h3_image_sequence(net, types.SimpleNamespace(), ["flux"])

    assert families == ["flux"]
    assert calls == []


def test_ignores_other_models():
    calls = []
    attention.install_attention = lambda net, cfg: calls.append("attn")
    attention.install_sequence = lambda net, cfg: calls.append("seq")

    families = backend.install_h3_image_sequence(OtherModel(), types.SimpleNamespace(), [])

    assert families == []
    assert calls == []
