"""Надёжное завершение задач и освобождение VRAM (Cancel в ComfyUI, «Unload models», зависшие ranks).

Исходный runtime.Session не меняется: install() (вызывается из дописанных строк __init__.py и при импорте
wan_runtime) подменяет на классе четыре метода безопасными версиями, исходные сохраняются в ORIGINAL.

Что исправляется:
  * дренаж после Cancel ждал ответ всех ranks без срока — один зависший rank (NCCL) навсегда блокировал
    все последующие задачи и держал VRAM. Теперь срок (POWERSHARD_DRAIN_TIMEOUT_S, по умолчанию
    max(600 с, 3 x самый долгий шаг)), после него workers принудительно завершаются;
  * повторный Cancel, пока идёт дренаж, = принудительный сброс (workers перезапустятся в следующей задаче);
  * close(): shutdown -> SIGTERM группе процесса -> SIGKILL, никогда не бросает исключение посреди
    остановки (раньше TimeoutExpired оставлял живые ranks, следующий старт получал OOM/ошибку NCCL);
  * comfy.model_management.unload_all_models (кнопка «Unload models» / «Free model and node cache»,
    восстановление после OOM, --disable-smart-memory) завершает все PowerShard workers — VRAM всех GPU
    реально освобождается, а не только «паркуется».
"""
import json
import os
import signal
import subprocess
import threading
import time
import warnings

ORIGINAL = {}
_FLAG_LOCK = threading.RLock()
_INSTALLED = {"session": False, "comfy": False}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# ------------------------------------------------------------------ processes
def _alive(proc):
    try:
        return proc.poll() is None
    except Exception:
        return False


def _signal_group(proc, sig):
    if not _alive(proc) or not isinstance(getattr(proc, "pid", None), int):
        return
    try:
        # Workers стартуют с start_new_session=True: pgid == pid, сигнал получают и дочерние процессы.
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _wait_all(processes, seconds):
    deadline = time.monotonic() + seconds
    for proc in processes:
        wait = getattr(proc, "wait", None)
        if wait is None:
            continue
        try:
            wait(timeout=max(.01, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, OSError):
            pass
    return [p for p in processes if _alive(p) and hasattr(p, "wait")]


def reap(processes, grace=None):
    """Остановить ranks: shutdown -> SIGTERM (группа) -> SIGKILL. Не бросает исключений."""
    processes = list(processes)
    if not processes:
        return []
    grace = _env_float("POWERSHARD_SHUTDOWN_GRACE_S", 3) if grace is None else grace
    for proc in processes:
        if _alive(proc):
            try:
                proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError, AttributeError):
                pass
    alive = _wait_all(processes, grace)
    for proc in alive:
        _signal_group(proc, signal.SIGTERM)
    alive = _wait_all(alive, 5)
    for proc in alive:
        _signal_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    alive = _wait_all(alive, 30)
    if alive:
        warnings.warn("PowerShard: ranks не завершились после SIGKILL (pid " + ", ".join(str(p.pid) for p in alive)
                      + "); VRAM освободится, когда драйвер завершит процессы")
    for proc in processes:
        for stream in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
            try:
                if stream:
                    stream.close()
            except (OSError, ValueError):
                pass
    return alive


def prune_reports(directory, keep=None):
    """Логи/progress/history ranks копятся с каждым стартом workers: оставить новейшие (POWERSHARD_KEEP_REPORTS)."""
    from pathlib import Path
    keep = int(_env_float("POWERSHARD_KEEP_REPORTS", 300)) if keep is None else keep
    try:
        files = [p for p in Path(directory).iterdir() if p.is_file() and p.name.startswith(("powershard-", "run-"))]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


# ------------------------------------------------------------------- session
def _abort(session):
    return session.__dict__.setdefault("_ps_abort", threading.Event())  # setdefault атомарен


def request_abort(session):
    """Прервать идущий дренаж (workers будут остановлены)."""
    with _FLAG_LOCK:
        if not session._drain_done.is_set():
            _abort(session).set()


def drain_timeout(session):
    explicit = os.environ.get("POWERSHARD_DRAIN_TIMEOUT_S")
    if explicit:
        return max(1., _env_float("POWERSHARD_DRAIN_TIMEOUT_S", 600))
    steps = [h.get("ipc_and_forward_s") or h.get("elapsed_s") or 0 for h in session.history[-200:]]
    # Первый шаг (прогрев, CPU offload) может быть долгим: нижняя граница 20 мин; дальше — 3 x самый долгий шаг.
    return max(1200., 3. * max([0.] + [float(s) for s in steps]))


def _owns_phase_lock():
    from . import runtime
    return getattr(runtime._PHASE_LOCK, "_is_owned", lambda: False)()


def safe_close(self, keep_stage=False):
    from . import runtime
    request_abort(self)  # дренаж (если идёт) отпустит lock в течение 50 мс
    with runtime._PHASE_LOCK, self.lock:
        processes, self.processes = list(self.processes), []
        reap(processes)
        try:
            ORIGINAL["close"](self, keep_stage)
        except BaseException as error:  # закрытие логов/каталогов не должно скрывать исходную ошибку
            warnings.warn(f"PowerShard close: {error}")
            self.path, self.sequence, self._wait_state = None, 0, None
            if runtime._ACTIVE is self:
                runtime._ACTIVE = None
    drain = self.__dict__.get("_ps_drain_thread")
    if not self._drain_done.is_set() and drain is not threading.current_thread() and not _owns_phase_lock():
        self._drain_done.wait(60)  # прерванный дренаж завершает свой except: следующая задача не застанет его
    with _FLAG_LOCK:
        if self._drain_done.is_set():
            _abort(self).clear()  # abort, пришедший без дренажа, не должен убить будущий штатный дренаж


def request_step_cancel(session, sequence):
    """Файл `<session_dir>/cancel` с номером команды: workers проверяют его перед каждым блоком (powershard.cancel)."""
    path = getattr(session, "path", None)
    if path is None:
        return
    try:
        target = os.path.join(str(path), "cancel")
        temporary = target + ".tmp"
        with open(temporary, "w") as f:
            f.write(str(sequence))
        os.replace(temporary, target)
    except OSError as error:
        warnings.warn(f"PowerShard: не удалось запросить прерывание шага ({error}); дождёмся конца шага")


def safe_drain(self, sequence, staging, out):
    """Как Session._drain_cancelled, но с крайним сроком и возможностью прервать."""
    import shutil
    from . import runtime
    abort = _abort(self)
    self.__dict__["_ps_drain_thread"] = threading.current_thread()
    timer = {"start": None, "limit": drain_timeout(self)}
    request_step_cancel(self, sequence)  # workers PowerShard (wan_worker) прервут шаг между блоками

    def guard():
        if abort.is_set():
            raise InterruptedError("дренаж прерван (повторный Cancel / выгрузка моделей)")
        if timer["start"] is not None and time.monotonic() - timer["start"] > timer["limit"]:
            raise TimeoutError(f"ranks не ответили за {timer['limit']:.0f} с после Cancel")

    try:
        # Lock держит поток sampler до выхода из sampling; ждём его, но тоже с проверкой abort.
        while not runtime._PHASE_LOCK.acquire(timeout=.05):
            guard()
        try:
            with self.lock:
                timer["start"] = time.monotonic()
                response = self._wait(sequence, guard)
                self.history[-1].update(command="cancelled_rpc_drain", status="DISCARDED_AFTER_INTERRUPT")
                self.last_memory = max(r.get("metrics", {}).get("memory", {}).get("allocated", 0) for r in response)
                self.control("end_run")
                self.idle()
        finally:
            runtime._PHASE_LOCK.release()
    except BaseException as error:
        warnings.warn(f"PowerShard: дренаж после Cancel не завершился штатно ({error}); workers остановлены, "
                      "VRAM освобождена, следующая задача загрузит модель заново")
        try:
            with runtime._PHASE_LOCK, self.lock:
                processes, self.processes = list(self.processes), []
                reap(processes, grace=0)
                ORIGINAL["close"](self, False)
        except BaseException as close_error:
            warnings.warn(f"PowerShard close после дренажа: {close_error}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
        with _FLAG_LOCK:
            abort.clear()
            self._drain_done.set()


def safe_wait_for_drain(self, cancel=None):
    """Ожидание дренажа; Cancel во время ожидания = принудительный сброс workers."""
    from .runtime import is_user_interrupt
    try:
        while not self._drain_done.wait(.05):
            if cancel is not None:
                cancel()
    except BaseException as error:
        if is_user_interrupt(error) and not self._drain_done.is_set():
            warnings.warn("PowerShard: повторный Cancel во время дренажа — workers будут остановлены")
            request_abort(self)
        raise


def _comfy_cancel():
    try:
        import comfy.model_management as mm
        return mm.throw_exception_if_processing_interrupted
    except Exception:
        return None


def wait_drain_bounded(session):
    """Для путей выгрузки ComfyUI: дождаться дренажа (он ограничен drain_timeout), реагируя на Cancel."""
    if session._drain_done.is_set():
        return
    if _owns_phase_lock():
        return  # дренаж ждёт этот lock: ожидание здесь было бы взаимной блокировкой
    session.wait_for_drain(_comfy_cancel())  # Cancel во время ожидания = принудительный сброс workers


def safe_wait_for_pending_phase(cancel=None):
    """Как runtime.wait_for_pending_phase, но ждёт дренаж ЛЮБОЙ session (не только _ACTIVE: close() его сбрасывает)."""
    from . import runtime
    sessions = list(runtime._SESSIONS)
    active = runtime._ACTIVE
    if active is not None and active not in sessions:
        sessions.append(active)
    for session in sessions:
        done = getattr(session, "_drain_done", None)
        if done is not None and not done.is_set():
            session.wait_for_drain(cancel)


def release_sessions(reason="unload_all_models"):
    """Остановить все PowerShard workers (все роли)."""
    from . import runtime
    sessions = [s for s in list(runtime._SESSIONS) if s.processes or not s._drain_done.is_set()]
    for session in sessions:  # сначала прервать все дренажи: иначе close() одной ждёт phase lock дренажа другой
        request_abort(session)
    for session in sessions:
        try:
            session.close()
        except BaseException as error:
            warnings.warn(f"PowerShard {reason}: {error}")


def install():
    """Идемпотентно: безопасные методы Session + обёртка comfy unload_all_models."""
    from . import runtime
    with _FLAG_LOCK:
        if not _INSTALLED["session"]:
            for name, function in (("close", safe_close), ("_drain_cancelled", safe_drain),
                                   ("wait_for_drain", safe_wait_for_drain)):
                ORIGINAL[name] = getattr(runtime.Session, name)
                setattr(runtime.Session, name, function)
            # Session.call/start и spectrum_outer_sample находят функцию через атрибут модуля runtime.
            ORIGINAL["wait_for_pending_phase"] = runtime.wait_for_pending_phase
            runtime.wait_for_pending_phase = safe_wait_for_pending_phase
            _INSTALLED["session"] = True
    install_comfy_hooks()


def settle_deferred_sessions():
    """После каждой задачи: отложенное (high-noise MoE) удержание VRAM не переживает задачу."""
    from . import runtime
    for session in list(runtime._SESSIONS):
        settle = getattr(session, "settle_deferred", None)
        if settle is not None:
            try:
                settle()
            except BaseException as error:
                warnings.warn(f"PowerShard: освобождение после задачи: {error}")


def _hook_prompt_end():
    try:
        import execution
    except Exception:
        return
    executor = getattr(execution, "PromptExecutor", None)
    original = getattr(executor, "execute", None)
    if original is None or getattr(original, "_powershard", False):
        return

    def execute(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            settle_deferred_sessions()
    execute._powershard = True
    execute.__wrapped__ = original
    executor.execute = execute


def install_comfy_hooks():
    try:
        import comfy.model_management as mm
    except Exception:  # вне ComfyUI (тесты, worker) — нечего оборачивать
        return
    with _FLAG_LOCK:
        if _INSTALLED["comfy"] or getattr(mm.unload_all_models, "_powershard", False):
            _INSTALLED["comfy"] = True
            return
        _hook_prompt_end()
        original = mm.unload_all_models

        def unload_all_models(*args, **kwargs):
            # Сначала PowerShard (прервать дренажи, остановить workers), затем ComfyUI: иначе его free_memory
            # ждал бы дренаж в partially_unload.
            try:
                release_sessions()
            except Exception as error:
                warnings.warn(f"PowerShard unload_all_models: {error}")
            return original(*args, **kwargs)
        unload_all_models._powershard = True
        unload_all_models.__wrapped__ = original
        mm.unload_all_models = unload_all_models
        _INSTALLED["comfy"] = True


# ------------------------------------------------------- sampler boundary (host)
def boundary_outer_sample(executor, *args, **kwargs):
    """outer_sample wrapper Wan/LTX: сброс Block Cache в начале задачи и отметка неудачи (Cancel/ошибка).

    Стоит ВНУТРИ исходного spectrum_outer_sample, поэтому флаг виден его finally -> finish_sampling():
    прерванный high-noise проход MoE больше не оставляет оба эксперта в VRAM «до low-noise прохода».
    """
    guider = executor.class_obj
    session = guider.model_patcher.session
    options = (getattr(guider, "model_options", None) or {}).get("transformer_options", {})
    holder = options.get("powershard_block_cache")
    if holder is not None and hasattr(holder, "begin_run"):
        holder.begin_run()
    session.__dict__["_ps_run_failed"] = False
    try:
        return executor(*args, **kwargs)
    except BaseException:
        session.__dict__["_ps_run_failed"] = True
        raise


boundary_outer_sample._powershard_host_safe = True
BOUNDARY_KEY = "powershard_boundary"


class BoundaryPatcherMixin:
    """Для WanPatcher / LTXPatcher: boundary wrapper и выгрузка, дожидающаяся дренажа."""

    def _install_boundary(self):
        if boundary_outer_sample not in self.get_wrappers("outer_sample", BOUNDARY_KEY):
            self.add_wrapper_with_key("outer_sample", BOUNDARY_KEY, boundary_outer_sample)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        wait_drain_bounded(self.session)
        return super().partially_unload(device_to, memory_to_free, force_patch_weights)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        wait_drain_bounded(self.session)
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)
