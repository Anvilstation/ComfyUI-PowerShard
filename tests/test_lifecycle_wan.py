"""Cancel / освобождение VRAM (powershard.lifecycle). Без torch/CUDA: реальные subprocess вместо ranks."""
import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import pytest
from powershard.config import DistributedConfig

HUNG_RANK = ("import signal, time\n"
             "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"   # зависший в NCCL rank: SIGTERM не помогает
             "time.sleep(120)\n")


def hung_process():
    return subprocess.Popen([sys.executable, "-c", HUNG_RANK], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, start_new_session=True)


@pytest.fixture
def lifecycle():
    from powershard import lifecycle
    lifecycle.install()
    return lifecycle


def make_session(tmp_path, keep=True):
    from powershard.runtime import Session
    session = Session(None, DistributedConfig(weight_placement="cpu", release_after_sampling=not keep), tmp_path, tmp_path)
    session.path = tmp_path / "workers"
    session.path.mkdir(exist_ok=True)
    return session


def test_reap_kills_rank_that_ignores_shutdown_and_sigterm(lifecycle):
    proc = hung_process()
    time.sleep(.3)
    started = time.monotonic()
    assert lifecycle.reap([proc], grace=.2) == []
    assert proc.poll() is not None and time.monotonic() - started < 20


def test_drain_has_deadline_and_releases_hung_ranks(lifecycle, tmp_path, monkeypatch):
    monkeypatch.setenv("POWERSHARD_DRAIN_TIMEOUT_S", "1")
    session = make_session(tmp_path)
    proc = hung_process()
    session.processes, session.responses = [proc], [queue.Queue()]
    session.sequence = 1
    session._drain_done.clear()
    staging, out = tmp_path / "in", tmp_path / "out"
    staging.mkdir()
    with pytest.warns(UserWarning, match="дренаж"):
        thread = threading.Thread(target=session._drain_cancelled, args=(1, staging, out))
        thread.start()
        assert session._drain_done.wait(30)
        thread.join(5)
    assert proc.poll() is not None and session.processes == [] and not staging.exists()


def test_second_cancel_while_draining_forces_reset(lifecycle, tmp_path, monkeypatch):
    monkeypatch.setenv("POWERSHARD_DRAIN_TIMEOUT_S", "3600")
    session = make_session(tmp_path)
    proc = hung_process()
    session.processes, session.responses = [proc], [queue.Queue()]
    session._drain_done.clear()
    staging, out = tmp_path / "in", tmp_path / "out"
    staging.mkdir()
    thread = threading.Thread(target=session._drain_cancelled, args=(1, staging, out))
    thread.start()

    def cancel():
        raise InterruptedError("второй Cancel")
    with pytest.warns(UserWarning):
        with pytest.raises(InterruptedError):
            session.wait_for_drain(cancel)
        assert session._drain_done.wait(30)
        thread.join(5)
    assert proc.poll() is not None and session.processes == []


def test_close_never_raises_and_resets_state(lifecycle, tmp_path):
    from powershard import runtime
    session = make_session(tmp_path, keep=False)
    session.processes = [hung_process(), hung_process()]
    runtime._ACTIVE = session
    session.close()
    assert session.processes == [] and session.path is None and runtime._ACTIVE is None


def wan_session(tmp_path, keep):
    from powershard.wan_runtime import WanSession
    session = WanSession(None, DistributedConfig(weight_placement="cpu", release_after_sampling=not keep), tmp_path,
                         tmp_path, role_options={"experts": {"high": "h", "low": "l"}})
    session.processes = [SimpleNamespace(poll=lambda: None)]
    calls = []
    session.control = lambda command: calls.append(command)
    session.close = lambda **kw: calls.append("close")
    session.idle = lambda: calls.append("idle")
    return session, calls


@pytest.mark.parametrize("keep", [True, False])
def test_interrupted_high_noise_pass_is_not_deferred(lifecycle, tmp_path, keep):
    session, calls = wan_session(tmp_path, keep)
    session.note_expert("high")
    session.finish_sampling()                       # успешный high-only проход: ждём low (VRAM остаётся)
    assert calls == ["end_run"] and session._wan_deferred
    calls.clear()
    session.note_expert("high")
    session.__dict__["_ps_run_failed"] = True       # Cancel внутри high-noise прохода
    session.finish_sampling()
    assert calls == ["end_run", "idle" if keep else "close"] and not session._wan_deferred
    assert session._wan_run_slots == set()


def test_prompt_end_settles_deferred_session(lifecycle, tmp_path, monkeypatch):
    from powershard import runtime
    session, calls = wan_session(tmp_path, keep=False)
    session.note_expert("high")
    session.finish_sampling()
    calls.clear()
    monkeypatch.setattr(runtime, "_SESSIONS", [session])
    lifecycle.settle_deferred_sessions()            # задача закончилась без low-noise прохода
    assert calls == ["end_run", "close"] and not session._wan_deferred


def test_drain_clears_stale_expert_slots(lifecycle, tmp_path):
    session, calls = wan_session(tmp_path, keep=True)
    session.note_expert("high")
    session._drain_done.clear()
    session.finish_sampling()
    session._drain_done.set()
    assert session._wan_run_slots == set() and calls == []


def test_boundary_wrapper_marks_failure_and_resets_block_cache(lifecycle):
    from powershard.accel import BlockCacheHolder
    holder = BlockCacheHolder(.1, .9, .1, 2, 1)
    holder.last_sigma, holder.steps = .3, 7
    session = SimpleNamespace()
    guider = SimpleNamespace(model_patcher=SimpleNamespace(session=session),
                             model_options={"transformer_options": {"powershard_block_cache": holder}})

    def failing(*args, **kwargs):
        raise InterruptedError()
    with pytest.raises(InterruptedError):
        lifecycle.boundary_outer_sample(_Executor(guider, failing))
    assert session._ps_run_failed and holder.last_sigma is None and holder.steps == 0
    assert lifecycle.boundary_outer_sample(_Executor(guider, lambda: "ok")) == "ok"
    assert session._ps_run_failed is False


class _Executor:
    def __init__(self, guider, function):
        self.class_obj, self.function = guider, function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def test_unload_all_models_releases_every_session(lifecycle, tmp_path, monkeypatch):
    from powershard import runtime
    released = []
    sessions = [SimpleNamespace(processes=[1], _drain_done=threading.Event(), close=lambda: released.append(1)),
                SimpleNamespace(processes=[], _drain_done=threading.Event(), close=lambda: released.append(2))]
    for s in sessions:
        s._drain_done.set()
    monkeypatch.setattr(runtime, "_SESSIONS", sessions)
    lifecycle.release_sessions()
    assert released == [1]


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_reap_on_already_exited_process(lifecycle):
    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    proc.wait(10)
    assert lifecycle.reap([proc]) == []


def test_request_abort_on_fresh_session_does_not_deadlock(lifecycle):
    session = SimpleNamespace(_drain_done=threading.Event())   # дренаж только что начат, Event abort ещё не создан
    done = threading.Event()
    threading.Thread(target=lambda: (lifecycle.request_abort(session), done.set()), daemon=True).start()
    assert done.wait(5) and session._ps_abort.is_set()


def test_abort_without_drain_is_cleared_by_close(lifecycle, tmp_path):
    session = make_session(tmp_path)
    session._drain_done.clear()
    lifecycle.request_abort(session)
    session._drain_done.set()                         # дренаж так и не стартовал (runtime.py: thread не создан)
    session.close()
    assert not session._ps_abort.is_set()             # следующий штатный дренаж не будет убит


def test_cancel_file_reaches_worker_check(lifecycle, tmp_path):
    """Cancel: host пишет <session>/cancel с номером команды, worker видит его только для этой команды."""
    from powershard import cancel
    session = SimpleNamespace(path=tmp_path)
    cancel.configure(str(tmp_path), None)
    cancel.begin(7)
    assert not cancel.requested_locally()
    lifecycle.request_step_cancel(session, 6)            # старая команда — не прерывает текущую
    assert not cancel.requested_locally()
    lifecycle.request_step_cancel(session, 7)
    assert cancel.requested_locally()
    cancel.end()
    assert not cancel.requested_locally()
