"""Host-side контракт Wan: native ComfyUI sampler -> proxy -> subprocess worker (CPU, без FSDP).

Проверяет MoE high->low без парковки между проходами, clones одной модели,
выбор эксперта по boundary и совпадение с native ComfyUI sampling.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import pytest

torch = pytest.importorskip("torch")
from wan_reference import import_comfy, tiny_config, native_model, save_native  # noqa: E402

WORKER = Path(__file__).with_name("wan_cpu_contract_worker.py")


@pytest.fixture(scope="module")
def wan(request):
    path = request.config.getoption("--comfy")
    if path is None:
        pytest.skip("NOT_RUN: передайте --comfy для Wan pipeline tests")
    return import_comfy(path)


def cpu_start(self, cancel=None):
    from powershard import runtime
    if self.running:
        if runtime._ACTIVE is not None and runtime._ACTIVE is not self:
            runtime._ACTIVE.deactivate()
        runtime._ACTIVE = self
        self.idle_on_cpu = False
        return
    self.close(keep_stage=True)
    self.path = Path(tempfile.mkdtemp(prefix="powershard-wan-test-"))
    self.report_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(comfy_path=self.comfy_path, role_options=self.role_options, config=self.config.to_dict(),
                    patch_fingerprint=self.patch.fingerprint())
    (self.path / "settings.json").write_text(json.dumps(settings))
    log = (self.report_dir / (self.path.name + ".log")).open("w")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    proc = subprocess.Popen([sys.executable, str(WORKER), str(self.path / "settings.json")], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env)
    self.processes, self.logs, self.responses = [proc], [log], [queue.Queue()]
    threading.Thread(target=self._reader, args=(proc.stdout, self.responses[0]), daemon=True).start()
    try:
        self._wait(0, cancel)
    except BaseException:
        self.close()
        raise
    runtime._ACTIVE = self


@pytest.fixture
def experts(wan, tmp_path, monkeypatch):
    config = tiny_config()
    high = save_native(native_model(wan, config, seed=1), tmp_path / "high.safetensors")
    low = save_native(native_model(wan, config, seed=2), tmp_path / "low.safetensors", prefix="model.diffusion_model.")
    monkeypatch.setattr("powershard.devices.visible_inventory",
                        lambda: [dict(user_id=str(i), uuid=f"GPU-{i}", name="CPU mock", total_memory=0) for i in range(2)])
    monkeypatch.setattr("powershard.wan_runtime.WanSession.start", cpu_start)
    return dict(high=str(high), low=str(low)), tmp_path


def load(experts, tmp_path, keep=True, **options):
    from powershard.config import DistributedConfig
    from powershard.wan_adapter import load_wan
    from powershard.wan_config import WanOptions
    config = DistributedConfig(attention_backend="sdpa", sequence_mode="ulysses", release_after_sampling=not keep)
    base = load_wan(experts, config, WanOptions(**options), report_dir=tmp_path / "reports", boundary=.875)
    base.load_device = torch.device("cpu")
    return base


def sample(model, latent, noise, start, last, positive, negative, full):
    import comfy.sample
    with torch.no_grad():
        return comfy.sample.sample(model, noise, 4, 2.0, "euler", "simple", positive, negative, latent,
                                   start_step=start, last_step=last, force_full_denoise=full, seed=7, disable_pbar=True)


def conds():
    g = torch.Generator().manual_seed(11)
    return [[torch.randn((1, 5, 32), generator=g), {}]], [[torch.randn((1, 5, 32), generator=g) * .1, {}]]


def native_patcher(path):
    import comfy.sd
    return comfy.sd.load_diffusion_model(path)


@pytest.mark.parametrize("keep", [True, False])
def test_moe_two_samplers_keep_workers_between_passes(experts, keep):
    import comfy.sample
    paths, tmp_path = experts
    base = load(paths, tmp_path, keep=keep)
    high, low = base.with_expert("high"), base.with_expert("low")
    session = high.session
    assert low.session is session and high.is_clone(low) and low.is_clone(high)
    positive, negative = conds()
    latent = torch.zeros((1, 16, 2, 4, 4))
    noise = comfy.sample.prepare_noise(latent, 7)
    try:
        mid = sample(high, latent, noise, 0, 2, positive, negative, False)
        assert session.running, "high-noise pass не должен закрывать workers MoE"
        assert not session.idle_on_cpu, "high-noise pass не должен парковать workers"
        pid = session.processes[0].pid
        out = sample(low, mid, torch.zeros_like(noise), 2, 10000, positive, negative, True)
        if keep:
            assert session.running and session.idle_on_cpu and session.processes[0].pid == pid
        else:
            assert not session.running
        used = {r["metrics"]["expert"] for h in session.history for r in h.get("ranks", []) or [] if r and r.get("metrics")}
        assert used == {"high", "low"}
        ref_high, ref_low = native_patcher(paths["high"]), native_patcher(paths["low"])
        ref_mid = sample(ref_high, latent, noise, 0, 2, positive, negative, False)
        ref_out = sample(ref_low, ref_mid, torch.zeros_like(noise), 2, 10000, positive, negative, True)
        torch.testing.assert_close(mid, ref_mid, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
    finally:
        session.close()


def test_moe_single_sampler_switches_by_boundary(experts):
    import comfy.sample
    paths, tmp_path = experts
    base = load(paths, tmp_path)
    moe = base.with_expert("moe", .875)
    session = moe.session
    positive, negative = conds()
    latent = torch.zeros((1, 16, 2, 4, 4))
    try:
        start = len(session.history)
        sample(moe, latent, comfy.sample.prepare_noise(latent, 3), None, None, positive, negative, False)
        used = [r["metrics"]["expert"] for h in session.history[start:] for r in h.get("ranks", []) or []
                if r and r.get("metrics")]
        assert used[0] == "high" and used[-1] == "low" and set(used) == {"high", "low"}
        assert session.idle_on_cpu
    finally:
        session.close()


def test_native_lora_and_patch_nodes_rejected(experts):
    paths, tmp_path = experts
    base = load(paths, tmp_path)
    high = base.with_expert("high")
    try:
        with pytest.raises(ValueError, match="PowerShard Wan LoRA"):
            high.add_patches({"diffusion_model.blocks.0.self_attn.q.weight": (torch.zeros(1),)})
        with pytest.raises(ValueError, match="Wan"):
            high.with_h3_patch(None)
        with pytest.raises(ValueError, match="H3"):
            high.with_spectrum(None)
    finally:
        high.session.close()


def test_lora_spec_reaches_worker_plan(experts, tmp_path):
    from safetensors.torch import save_file
    from powershard.wan_config import WanLoraSpec
    from powershard.config import DistributedConfig
    from powershard.wan_adapter import load_wan
    from powershard.wan_config import WanOptions
    paths, _ = experts
    lora = tmp_path / "lora.safetensors"
    save_file({"diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.zeros((256, 4), dtype=torch.float16),
               "diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.zeros((4, 256), dtype=torch.float16)}, str(lora))
    base = load_wan(paths, DistributedConfig(attention_backend="sdpa"), WanOptions(),
                    (WanLoraSpec(str(lora), .7, "low_noise"),), tmp_path / "r")
    try:
        plan = base.session.role_options["loras"]
        assert list(plan) == ["low"] and plan["low"][0]["strength"] == .7
    finally:
        base.session.close()
    bad = tmp_path / "bad.safetensors"
    save_file({"diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.zeros((128, 4), dtype=torch.float16),
               "diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.zeros((4, 256), dtype=torch.float16)}, str(bad))
    with pytest.raises(ValueError, match="форма"):
        load_wan(paths, DistributedConfig(attention_backend="sdpa"), WanOptions(), (WanLoraSpec(str(bad)),), tmp_path / "r")
