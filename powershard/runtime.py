"""N постоянных subprocess, приватный FileStore, pipes, без Ray/fork CUDA."""
import atexit
from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import time
from .config import DistributedConfig
from .patch_config import H3PatchConfig
from .wire import StagedTensor

_SESSIONS = weakref.WeakSet()
_ACTIVE = None
_PHASE_LOCK = threading.RLock()


def resolve_gpus(ids):
    from .devices import resolve_gpu_selection
    return [d["uuid"] for d in resolve_gpu_selection(ids)]


class Session:
    def __init__(self, checkpoint, config, comfy_path, report_dir=None, probe_only=False, patch=None, role="h3", role_options=None):
        self.checkpoint, self.config = checkpoint, config
        self.comfy_path = str(Path(comfy_path).resolve())
        self.report_dir = Path(report_dir or Path(tempfile.gettempdir()) / "powershard-reports")
        self.probe_only = probe_only
        self.patch = patch or H3PatchConfig()
        if role not in ("h3","qwen"):
            raise ValueError("Неизвестная worker role")
        self.role,self.role_options=role,dict(role_options or {})
        self.idle_on_cpu=False
        self.processes, self.logs, self.responses = [], [], []
        self.path = None
        self.lock = threading.RLock()
        self.sequence = 0
        self.history = []
        self.last_memory = 0
        self.selected_devices = []
        self.attention_policy = None
        self._provider_stamp = None
        self._request_stamp = None
        self.sampling_context = None
        self.stage = None
        self._stage_data = {}
        _SESSIONS.add(self)

    def with_patch(self, patch):
        return Session(self.checkpoint, self.config, self.comfy_path, self.report_dir, self.probe_only, patch,self.role,self.role_options)

    def with_spectrum(self, spectrum):
        return Session(self.checkpoint,self.config,self.comfy_path,self.report_dir,self.probe_only,self.patch,
                       self.role,dict(self.role_options,spectrum=spectrum.to_dict()))

    @property
    def running(self):
        return bool(self.processes) and all(p.poll() is None for p in self.processes)

    def _reader(self, stream, mailbox):
        try:
            for line in stream:
                try:
                    mailbox.put(json.loads(line))
                except json.JSONDecodeError:
                    mailbox.put({"error": "Повреждённый worker protocol", "line": line[:1000]})
        finally:
            mailbox.put({"error": "Worker завершился без ответа"})

    def start(self, cancel=None):
        global _ACTIVE
        with _PHASE_LOCK, self.lock:
            from .web_api import provider_stamp
            stamp = provider_stamp()
            checkpoint_stamp = (self.checkpoint,Path(self.checkpoint).stat().st_size,Path(self.checkpoint).stat().st_mtime_ns) if self.checkpoint else None
            request_stamp = (json.dumps(self.config.to_dict(),sort_keys=True),self.patch.fingerprint(),checkpoint_stamp,os.environ.get("CUDA_VISIBLE_DEVICES"),self.role,json.dumps(self.role_options,sort_keys=True))
            if self.running and self._provider_stamp == stamp and self._request_stamp == request_stamp:
                if _ACTIVE is not None and _ACTIVE is not self:
                    _ACTIVE.deactivate()
                _ACTIVE=self
                self.idle_on_cpu=False
                return
            # allow_unverified сохранён для старых workflows, но больше не
            # блокирует испытания. Admission — реальные capabilities/preflight.
            if _ACTIVE is not None and _ACTIVE is not self:
                _ACTIVE.deactivate()
            self.close()
            from .devices import resolve_gpu_selection
            self.selected_devices = resolve_gpu_selection(self.config.gpu_ids)
            uuids = [d["uuid"] for d in self.selected_devices]
            # Probes в отдельных процессах, до NCCL/FSDP: единая policy для ranks.
            from .attention_policy import prepare_policy
            geometry=None
            if self.role=="qwen":
                from .qwen import infer_qwen_config
                from .checkpoint import Checkpoint
                shape=infer_qwen_config(Checkpoint(self.checkpoint).tensors)
                geometry={"head_dim":shape["head_dim"],"heads":shape["num_attention_heads"],
                          "kv_heads":shape.get("num_key_value_heads",shape["num_attention_heads"])}
            self.attention_policy = prepare_policy(self.config, self.selected_devices, self.checkpoint if self.role=="h3" else None,geometry=geometry)
            self._provider_stamp = stamp
            self._request_stamp = request_stamp
            self.report_dir.mkdir(parents=True, exist_ok=True)
            self.path = Path(tempfile.mkdtemp(prefix="powershard-"))
            settings = {"checkpoint": self.checkpoint, "config": self.config.to_dict(), "comfy_path": self.comfy_path,
                        "role":self.role,"role_options":self.role_options,
                        "patch": self.patch.to_dict(), "gpu_uuids": uuids,
                        "selected_devices": self.selected_devices, "attention_policy": self.attention_policy,
                        "session_dir": str(self.path), "report_dir": str(self.report_dir.resolve()), "probe_only": self.probe_only, "parent_pid": os.getpid()}
            from .attention_policy import policy_fingerprint
            self.fingerprint = policy_fingerprint(dict(config=self.config.to_dict(),devices=uuids,patch=self.patch.to_dict(),
                attention=self.attention_policy["fingerprint"],provider_stamp=stamp,
                checkpoint=checkpoint_stamp,role=self.role,role_options=self.role_options))
            settings["fingerprint"] = self.fingerprint
            if self.config.weight_placement == "ats":
                from .ats_probe import isolated_probe
                import warnings
                settings["ats_diagnostics"] = [isolated_probe(d,min(90,self.config.timeout_s)) for d in self.selected_devices]
                for report in settings["ats_diagnostics"]:
                    if report["status"] == "ERROR":
                        warnings.warn(f"ATS diagnostic failed; using ordinary CPUOffloadPolicy: {report}")
            (self.path / "settings.json").write_text(json.dumps(settings))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), self.comfy_path, env.get("PYTHONPATH", "")])
            env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
            # Не заменяем NCCL, allocator, P2P/SHM. Наследуем установленный стек.
            try:
                for rank in range(len(uuids)):
                    from .topology import numa_launch_prefix
                    prefix = numa_launch_prefix(uuids[rank], self.config.numa_policy)
                    logpath = self.report_dir / f"{self.path.name}-rank{rank}.log"
                    log = logpath.open("w")
                    mailbox = queue.Queue()
                    proc = subprocess.Popen(prefix + [sys.executable, "-u", "-m", "powershard.worker", str(self.path / "settings.json"), str(rank)],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                                            env=env, cwd=self.comfy_path, start_new_session=True)
                    self.processes.append(proc); self.logs.append(log); self.responses.append(mailbox)
                    threading.Thread(target=self._reader, args=(proc.stdout, mailbox), daemon=True).start()
                replies = self._wait(0, cancel)
                if not self.probe_only and any(r["preflight"].get("patch_fingerprint") != self.patch.fingerprint() for r in replies):
                    raise RuntimeError("Worker patch configuration отличается между ranks")
                if not self.probe_only and any(r["preflight"]["attention"]["policy"].get("fingerprint") != self.attention_policy["fingerprint"] for r in replies):
                    raise RuntimeError("Worker attention policy отличается между ranks")
                _ACTIVE = self
            except BaseException:
                self.close()
                raise

    def _wait(self, sequence, cancel=None):
        deadline = time.monotonic() + self.config.timeout_s
        pending = set(range(len(self.processes)))
        result = [None] * len(self.processes)
        while pending:
            if cancel is not None:
                cancel()  # ComfyUI interrupt exception
            if time.monotonic() > deadline:
                raise TimeoutError(f"PowerShard timeout на команде {sequence}; все свои workers будут завершены")
            for i in list(pending):
                try:
                    msg = self.responses[i].get(timeout=.02)
                except queue.Empty:
                    continue
                if "error" in msg:
                    raise RuntimeError(f"PowerShard rank {i}: {msg['error']}; логи: {self.report_dir}")
                if msg.get("sequence") != sequence:
                    raise RuntimeError("Нарушен порядок worker commands")
                result[i] = msg
                pending.remove(i)
        self.history.append({"sequence": sequence, "ranks": result})
        return result

    def stage_tensors(self, tensors):
        """Один раз за run пишет conditioning-тензоры в run-stage каталог.

        Файл ПЕРЕЗАПИСЫВАЕТСЯ целиком всеми накопленными тензорами: в run
        бывает несколько разных context (cond/uncond, изменение prompt) —
        частичная запись стёрла бы ранее staged ключи. Повторный вызов с тем
        же content-hash не дублирует данные.
        """
        with _PHASE_LOCK, self.lock:
            if self.stage is None:
                self.stage = Path(tempfile.mkdtemp(prefix="powershard-stage-"))
            updated = dict(self._stage_data)
            for name, tensor in tensors.items():
                if name not in self._stage_data:
                    updated[name] = tensor.detach().to("cpu").contiguous().clone()
            if len(updated) != len(self._stage_data):
                from safetensors.torch import save_file
                temporary = self.stage / "tensors.tmp.safetensors"
                save_file(updated, str(temporary))
                os.replace(temporary, self.stage / "tensors.safetensors")
                self._stage_data = updated
            return {k: StagedTensor(k, index=0) for k in self._stage_data}

    def stage_dir(self):
        with self.lock:
            return self.stage

    def clear_stage(self):
        with self.lock:
            if self.stage is not None:
                shutil.rmtree(self.stage, ignore_errors=True)
            self.stage = None
            self._stage_data.clear()

    def call(self, command, args, kwargs, cancel=None):
        from .wire import write_payload, read_payload
        with _PHASE_LOCK, self.lock:
            # Сначала сериализация+проверка patches, затем старт/collectives.
            staging = Path(tempfile.mkdtemp(prefix="powershard-input-"))
            try:
                serialization_start = time.perf_counter()
                write_payload(staging, {"args": args, "kwargs": kwargs})
                serialization_s = time.perf_counter()-serialization_start
                self.start(cancel)
                self.sequence += 1
                out = self.path / f"output-{self.sequence}"
                start = time.perf_counter()
                base = str(self.stage) if self.stage is not None else ""
                req = json.dumps({"sequence": self.sequence, "command": command,
                                  "input": str(staging), "output": str(out), "stage": base})
                for proc in self.processes:
                    proc.stdin.write(req + "\n"); proc.stdin.flush()
                response = self._wait(self.sequence, cancel)
                result = read_payload(out)
                shutil.rmtree(out)
                self.history[-1]["ipc_and_forward_s"] = time.perf_counter()-start
                self.history[-1]["input_serialization_s"] = serialization_s
                self.history[-1]["command"] = command
                self.last_memory = max(r.get("metrics", {}).get("memory", {}).get("allocated", 0) for r in response)
                return result
            except BaseException:
                self.close()
                raise
            finally:
                # run-stage НЕ удаляется здесь: он живёт до конца run
                # (close()). Шаговый staging удаляется как раньше.
                shutil.rmtree(staging, ignore_errors=True)

    def close(self):
        global _ACTIVE
        with _PHASE_LOCK, self.lock:
            processes, self.processes = self.processes, []
            for p in processes:
                if p.poll() is None:
                    try:
                        p.stdin.write(json.dumps({"command": "shutdown"})+"\n"); p.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
            deadline = time.monotonic() + 2
            for p in processes:
                try:
                    p.wait(timeout=max(.01, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(timeout=2)
                for s in (p.stdin, p.stdout):
                    if s:
                        s.close()
            for f in self.logs:
                f.close()
            self.logs, self.responses = [], []
            if self.path is not None:
                if self.history:
                    self.report_dir.mkdir(parents=True, exist_ok=True)
                    (self.report_dir / (self.path.name + ".json")).write_text(json.dumps(self.history, indent=2))
                shutil.rmtree(self.path, ignore_errors=True)
            if self.stage is not None:
                shutil.rmtree(self.stage, ignore_errors=True)
            self.stage = None
            self._stage_data = {}
            self.path = None
            self.sequence = 0
            self.last_memory = 0
            self.idle_on_cpu=False
            if _ACTIVE is self:
                _ACTIVE = None

    def deactivate(self):
        if self.role=="qwen" and self.role_options.get("idle_policy")=="cpu_shards" and self.running:
            self.idle()
        else:
            self.close()

    def idle(self):
        global _ACTIVE
        with _PHASE_LOCK, self.lock:
            if not self.running or self.idle_on_cpu:return
            if not self.config.cpu_offload:
                self.close();return
            self.sequence+=1
            request=json.dumps(dict(command="idle",sequence=self.sequence))
            try:
                for p in self.processes:p.stdin.write(request+"\n");p.stdin.flush()
                replies=self._wait(self.sequence)
                self.clear_stage()
                self.last_memory=max(r.get("phase_offload",{}).get("after",{}).get("allocated",0) for r in replies)
                self.idle_on_cpu=True
                if _ACTIVE is self:_ACTIVE=None
            except BaseException:
                self.close();raise

    def control(self,command):
        if command!="end_run":raise ValueError("Неизвестная session control command")
        with _PHASE_LOCK,self.lock:
            if not self.running:
                self.clear_stage()
                return
            self.sequence+=1
            try:
                for p in self.processes:
                    p.stdin.write(json.dumps(dict(command=command,sequence=self.sequence))+"\n");p.stdin.flush()
                replies=self._wait(self.sequence)
                self.history[-1]["command"]=command
                self.clear_stage()
                return replies
            except BaseException:
                self.close();raise

    def save_run_summary(self,context):
        self.report_dir.mkdir(parents=True,exist_ok=True)
        value=dict(run_id=context["run_id"],role=self.role,checkpoint=self.checkpoint,
            config=self.config.to_dict(),patch=self.patch.to_dict(),role_options=self.role_options,
            sampler_bridge=dict(eligible=context["eligible"],reason=context["reason"]),
            steps=context["steps"],sigmas=context["sigmas"],
            manifest=context.get("manifest",{}),
            sampling_wall_s=time.perf_counter()-context["started"],
            session_fingerprint=getattr(self,"fingerprint",None),
            history=self.history[context["history_start"]:],
            note="Sampling boundary only: text encoding/VAE/video encoding are separate phases; no raw prompt stored")
        (self.report_dir/("run-"+context["run_id"]+".json")).write_text(json.dumps(value,indent=2))


def release_all(preserve_idle_cpu=False):
    for s in list(_SESSIONS):
        if preserve_idle_cpu and s.role=="qwen" and s.idle_on_cpu:continue
        s.close()


atexit.register(release_all)
