"""Host-side Session для роли ``wan``: подкласс runtime.Session без изменения оригинала.

Переопределено только то, что в Session жёстко привязано к H3/Qwen: роль,
attention probes (геометрия Wan), модуль worker и отложенная парковка между
high- и low-noise sampler проходами одного MoE.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import queue
import warnings
from . import runtime
from .runtime import Session, _SESSIONS, wait_for_pending_phase
from .patch_config import H3PatchConfig


_POLICY_CACHE = {}


def _probe_all(provider, devices, head_dim, probe_heads):
    """Изолированные CUDA probes параллельно (по процессу на GPU и число heads), порядок результатов сохранён."""
    from concurrent.futures import ThreadPoolExecutor
    from .attention_policy import isolated_probe

    def per_device(device):  # на одном GPU — по очереди, GPU между собой — параллельно
        return [dict(isolated_probe(provider, device, head_dim, h, 90), probe_heads=h) for h in probe_heads]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        return [r for rows in pool.map(per_device, devices) for r in rows]


def wan_prepare_policy(config, devices, heads, head_dim):
    """Кэш на процесс ComfyUI: повторный старт той же session не запускает 6-12 probe-процессов заново.

    Ключ — GPU UUID, запрошенный backend/fallback/режим и геометрия heads; provider stamp
    (версии/файлы attention-сборок) входит в ключ, а worker всё равно сверяет identity provider.
    """
    from .web_api import provider_stamp
    key = json.dumps(dict(devices=[d["uuid"] for d in devices], requested=config.requested_attention,
                          fallback=config.allow_fallback, mode=config.sequence_mode, heads=heads, head_dim=head_dim,
                          provider=provider_stamp()), sort_keys=True, default=str)
    if key in _POLICY_CACHE:
        return json.loads(json.dumps(_POLICY_CACHE[key]))
    policy = _wan_prepare_policy(config, devices, heads, head_dim)
    if not policy.get("reason"):  # fallback (возможный ложный FAIL probe) не кэшируется: следующий старт проверит снова
        _POLICY_CACHE[key] = policy
    return policy


def _wan_prepare_policy(config, devices, heads, head_dim):
    """Как attention_policy.prepare_policy, но с геометрией Wan.

    Self-attention Ulysses использует ceil(H/world) heads, cross-attention — все H.
    Custom Volta kernels должны пройти ОБЕ формы, иначе вызов уйдёт в SDPA/math.
    """
    from .attention_policy import policy_fingerprint
    requested = config.requested_attention
    candidates = ["vllm_flash_attn", "flash_attn", "sdpa", "math"] if requested == "auto" else [requested]
    if config.allow_fallback:
        if requested == "flash_attn":
            candidates.append("vllm_flash_attn")
        candidates += ["sdpa", "math"]
    probe_heads = [heads]
    if config.sequence_mode == "ulysses" and len(devices) > 1:
        probe_heads = list(dict.fromkeys([heads, (heads + len(devices) - 1) // len(devices)]))
    reports = {}
    for provider in dict.fromkeys(candidates):
        reports[provider] = _probe_all(provider, devices, head_dim, probe_heads)
        if all(r["status"] == "PASS" for r in reports[provider]):
            policy = dict(requested=requested, effective=provider, probes=reports,
                          geometry=dict(head_dim=head_dim, heads=heads, dtype="float16", layout="BLHD",
                                        kv_heads=heads, certified_heads=probe_heads, gqa_certified=False, model="wan"),
                          devices=[d["uuid"] for d in devices], allow_fallback=config.allow_fallback,
                          reason=None if requested in (provider, "auto") else reports.get(requested),
                          provider_identities=[r["identity"] for r in reports[provider]])
            policy["fingerprint"] = policy_fingerprint(policy)
            if policy["reason"]:
                warnings.warn(f"Requested: {requested}; Effective: {provider}; Reason: "
                              + json.dumps(policy["reason"], ensure_ascii=False)[:2000])
            return policy
    raise RuntimeError("Нет проверенного attention пути на всех выбранных GPU: " + json.dumps(reports, ensure_ascii=False)[:8000])


def reusable_wan_session(checkpoint, config, comfy_path, report_dir=None, probe_only=False, patch=None, role_options=None):
    patch = patch or H3PatchConfig(enabled=True)
    options = json.loads(json.dumps(role_options or {}, sort_keys=True))
    directory = Path(report_dir or Path(tempfile.gettempdir()) / "powershard-reports")
    comfy = str(Path(comfy_path).resolve())
    if not config.release_after_sampling:
        wait_for_pending_phase(getattr(sys.modules.get("comfy.model_management"),
                                       "throw_exception_if_processing_interrupted", None))
    with runtime._PHASE_LOCK:
        # Одна session для всех MODEL выходов MoE loader: high/low/moe делят workers.
        for session in list(_SESSIONS):
            if (isinstance(session, WanSession) and session.checkpoint == checkpoint and session.config == config
                    and session.comfy_path == comfy and session.report_dir == directory
                    and session.probe_only == probe_only and session.patch == patch and session.role_options == options):
                return session
    return WanSession(checkpoint, config, comfy, directory, probe_only, patch, options)


class WanSession(Session):
    def __init__(self, checkpoint, config, comfy_path, report_dir=None, probe_only=False, patch=None, role_options=None):
        # Session.__init__ проверяет роль h3/qwen; роль устанавливается после.
        super().__init__(checkpoint, config, comfy_path, report_dir, probe_only, patch, "h3", role_options)
        # Отдельная роль для umT5: старт энкодера не закрывает генератор той же роли (только паркует).
        kind = (role_options or {}).get("kind")
        self.role = {"t5": "wan_t5", "native_te": "native_te", "ltx": "ltx", "h3q": "h3"}.get(kind, "wan")
        self.kind = kind
        self._wan_run_slots = set()
        self._wan_deferred = False

    # --- reuse/lifecycle -------------------------------------------------
    def with_patch(self, patch):
        return reusable_wan_session(self.checkpoint, self.config, self.comfy_path, self.report_dir,
                                    self.probe_only, patch, self.role_options)

    def with_spectrum(self, spectrum):
        if self.role_options.get("kind") == "h3q":
            return reusable_wan_session(self.checkpoint, self.config, self.comfy_path, self.report_dir, self.probe_only,
                                        self.patch, dict(self.role_options, spectrum=spectrum.to_dict()))
        raise ValueError("Spectrum реализован только для MiniMax H3; для Wan не поддерживается")

    @property
    def retains_weights(self):
        return not self.config.release_after_sampling

    @property
    def experts(self):
        return dict(self.role_options.get("experts", {}))

    def note_expert(self, slot):
        self._wan_run_slots.add(slot)

    def should_defer(self):
        """После прохода, использовавшего только high expert пары, ждём low-noise проход."""
        experts = self.experts
        return ("high" in experts and "low" in experts and self._wan_run_slots == {"high"} and self.running)

    def finish_sampling(self):
        if self.draining:
            return
        defer = self.should_defer()
        self._wan_run_slots = set()
        if defer:
            # Conditioning stage очищается, workers и VRAM-shards остаются для low-noise.
            self._wan_deferred = True
            if self.running:
                self.control("end_run")
            return
        self._wan_deferred = False
        super().finish_sampling()

    def deactivate(self):
        self._wan_deferred = False
        return super().deactivate()

    def close(self, keep_stage=False):
        # _wan_run_slots не сбрасывается: start() вызывает close(keep_stage=True) уже
        # после note_expert() первого forward. Слоты очищает finish_sampling().
        self._wan_deferred = False
        return super().close(keep_stage)

    # --- startup ---------------------------------------------------------
    def _wan_stamp(self):
        stamps = {}
        for slot, path in self.experts.items():
            st = Path(path).stat()  # только stat: start() вызывается на каждом RPC
            stamps[slot] = [str(path), st.st_size, st.st_mtime_ns]
        for slot, loras in self.role_options.get("loras", {}).items():
            for item in loras:
                st = Path(item["path"]).stat()
                stamps.setdefault("lora", []).append([slot, item["path"], st.st_size, st.st_mtime_ns])
        return (json.dumps(self.config.to_dict(), sort_keys=True), self.patch.fingerprint(),
                json.dumps(stamps, sort_keys=True), os.environ.get("CUDA_VISIBLE_DEVICES"), self.role,
                json.dumps(self.role_options, sort_keys=True))

    def start(self, cancel=None):
        self.wait_for_drain(cancel)
        wait_for_pending_phase(cancel)
        with runtime._PHASE_LOCK, self.lock:
            from .web_api import provider_stamp
            stamp = provider_stamp()
            request_stamp = self._wan_stamp()
            if self.running and self._provider_stamp == stamp and self.__dict__.get("_wan_request_stamp") == request_stamp:
                if runtime._ACTIVE is not None and runtime._ACTIVE is not self:
                    runtime._ACTIVE.deactivate()
                runtime._ACTIVE = self
                self.idle_on_cpu = False
                return
            if runtime._ACTIVE is not None and runtime._ACTIVE is not self:
                runtime._ACTIVE.deactivate()
            for other in list(_SESSIONS):
                if other is not self and other.role == self.role and other.running:
                    other.close()
            self.close(keep_stage=True)
            from .devices import resolve_gpu_selection
            from .wan_config import WanCheckpoint, same_geometry
            self.selected_devices = resolve_gpu_selection(self.config.gpu_ids)
            uuids = [d["uuid"] for d in self.selected_devices]
            if self.role in ("wan_t5", "native_te"):
                # Text encoders считаются native comfy attention (FP32 SDPA) — CUDA probes FP16 kernels не нужны.
                from .attention_policy import policy_fingerprint
                self.attention_policy = dict(requested="comfy_native", effective="comfy_native_fp32",
                                             devices=uuids, reason="text encoder: FP32 compute, native comfy attention")
                self.attention_policy["fingerprint"] = policy_fingerprint(self.attention_policy)
            elif self.role_options.get("kind") == "h3q":
                from .attention_policy import prepare_policy
                self.attention_policy = prepare_policy(self.config, self.selected_devices, self.checkpoint)
            elif self.role == "ltx":
                from .ltx_config import LTXCheckpoint, ltx_geometry
                geometry = ltx_geometry(LTXCheckpoint(self.experts["main"]).model_config())
                self.attention_policy = wan_prepare_policy(self.config, self.selected_devices, geometry["heads"],
                                                           geometry["head_dim"])
            else:
                geometries = {slot: WanCheckpoint(path, self.role_options.get("options", {}).get("model_type", "auto"))
                              .model_config() for slot, path in self.experts.items()}
                if not geometries:
                    raise ValueError("Wan session без экспертов")
                first = next(iter(geometries.values()))
                if any(not same_geometry(first, g) for g in geometries.values()):
                    raise ValueError("Эксперты MoE имеют разную геометрию Wan: " + json.dumps(geometries, default=str))
                self.attention_policy = wan_prepare_policy(self.config, self.selected_devices, first["num_heads"],
                                                           first["dim"] // first["num_heads"])
            self._provider_stamp = stamp
            self._wan_request_stamp = request_stamp
            self.report_dir.mkdir(parents=True, exist_ok=True)
            self.path = Path(tempfile.mkdtemp(prefix="powershard-wan-"))
            settings = {"checkpoint": self.checkpoint, "config": self.config.to_dict(), "comfy_path": self.comfy_path,
                        "role": "wan", "role_options": self.role_options, "patch": self.patch.to_dict(),
                        "gpu_uuids": uuids, "selected_devices": self.selected_devices,
                        "attention_policy": self.attention_policy, "session_dir": str(self.path),
                        "report_dir": str(self.report_dir.resolve()), "probe_only": self.probe_only,
                        "parent_pid": os.getpid()}
            from .attention_policy import policy_fingerprint
            self.fingerprint = policy_fingerprint(dict(config=self.config.to_dict(), devices=uuids, patch=self.patch.to_dict(),
                                                       attention=self.attention_policy["fingerprint"], provider_stamp=stamp,
                                                       request=request_stamp, role="wan"))
            settings["fingerprint"] = self.fingerprint
            (self.path / "settings.json").write_text(json.dumps(settings))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), self.comfy_path,
                                                 env.get("PYTHONPATH", "")])
            env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
            if self.config.weight_placement == "ats":
                from .ats_memory import ats_worker_environment
                env = ats_worker_environment(env)
            try:
                from .topology import numa_launch_prefix
                for rank in range(len(uuids)):
                    prefix = numa_launch_prefix(uuids[rank], self.config.numa_policy)
                    log = (self.report_dir / f"{self.path.name}-rank{rank}.log").open("w")
                    mailbox = queue.Queue()
                    proc = subprocess.Popen(prefix + [sys.executable, "-u", "-m", "powershard.wan_worker",
                                                      str(self.path / "settings.json"), str(rank)],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                                            env=env, cwd=self.comfy_path, start_new_session=True)
                    self.processes.append(proc)
                    self.logs.append(log)
                    self.responses.append(mailbox)
                    threading.Thread(target=self._reader, args=(proc.stdout, mailbox), daemon=True).start()
                replies = self._wait(0, cancel)
                if not self.probe_only and any(r["preflight"].get("patch_fingerprint") != self.patch.fingerprint() for r in replies):
                    raise RuntimeError("Worker patch configuration отличается между ranks")
                if not self.probe_only and any(r["preflight"]["attention"]["policy"].get("fingerprint") != self.attention_policy["fingerprint"]
                                               for r in replies):
                    raise RuntimeError("Worker attention policy отличается между ranks")
                if not self.probe_only and len({json.dumps(r["preflight"].get("wan_fingerprints"), sort_keys=True) for r in replies}) != 1:
                    raise RuntimeError("Wan expert/LoRA fingerprints отличаются между ranks")
                self.preflight = replies
                runtime._ACTIVE = self
            except BaseException:
                self.close()
                raise
