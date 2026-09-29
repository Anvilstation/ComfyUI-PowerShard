"""Опциональные измерения реального ComfyUI execution, без GPU работы при импорте.

Включение: POWERSHARD_BENCHMARK_DIR=/absolute/reports перед запуском ComfyUI.
Hook проверяет capabilities execution.py, без SHA/version gate. Async/subgraph node timings
помечаются неполными; планировщик и порядок исполнения не изменяются.
"""
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid


STAGES = {
    "PowerShardH3Loader": "generator_metadata_load",
    "PowerShardH3TextEncoder": "text_encoder_checkpoint_load",
    "PowerShardH3QwenLoader": "text_encoder_metadata_load",
    "MiniMaxH3ImageToVideo": "conditioning_text_image_encoding",
    "MiniMaxH3ReferenceToVideo": "conditioning_text_reference_encoding",
    "CLIPLoader": "text_encoder_checkpoint_load",
    "CLIPTextEncode": "text_encoding_including_lazy_load",
    "VAELoader": "vae_checkpoint_load",
    "SamplerCustomAdvanced": "denoising_node_including_load_and_release",
    "KSampler": "denoising_node_including_load_and_release",
    "VAEDecodeTiled": "video_vae_decode_including_lazy_load",
    "VAEDecode": "video_vae_decode_including_lazy_load",
    "VAEDecodeAudio": "audio_vae_decode_including_lazy_load",
    "PowerShardRelease": "worker_release",
    "SaveVideo": "video_audio_save",
}


def cuda_snapshot(reset=False):
    # Только устройства, уже имеющие host allocations. Не создавать контексты
    # на всех трёх GPU только ради измерения памяти host process.
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return []
    rows = []
    for i in range(torch.cuda.device_count()):
        if not torch.cuda.memory_allocated(i) and not torch.cuda.memory_reserved(i):
            continue
        torch.cuda.synchronize(i)
        rows.append(dict(device=i, allocated=torch.cuda.memory_allocated(i),
                         reserved=torch.cuda.memory_reserved(i),
                         peak_allocated=torch.cuda.max_memory_allocated(i),
                         peak_reserved=torch.cuda.max_memory_reserved(i)))
        if reset:
            torch.cuda.reset_peak_memory_stats(i)
    return rows


class GPUInventory:
    """GPU-wide память включает NCCL/внешние процессы; это sampled peak, не exact peak."""
    def __init__(self):
        self.stop = threading.Event()
        self.thread = None
        self.samples = []
        self.error = None
        self.stage = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free",
                                               "--format=csv,noheader,nounits"], text=True, timeout=3)
                rows = []
                for line in raw.strip().splitlines():
                    ident, used, free = [x.strip() for x in line.split(",")]
                    rows.append(dict(uuid=ident, used_bytes=int(used)*2**20, free_bytes=int(free)*2**20))
                self.samples.append(dict(monotonic_s=time.monotonic(), stage=self.stage, gpus=rows))
            except (OSError, ValueError, subprocess.SubprocessError) as e:
                self.error = str(e)
                return
            self.stop.wait(.5)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=4)


def install_if_requested():
    destination = os.environ.get("POWERSHARD_BENCHMARK_DIR")
    if not destination:
        return False
    execution = sys.modules.get("execution")
    if execution is None or not hasattr(execution, "PromptExecutor"):
        raise RuntimeError("Benchmark hook должен устанавливаться при загрузке custom nodes в ComfyUI")
    from .source_guard import verify_comfy
    verify_comfy(Path(execution.__file__).resolve().parent)
    return install(execution, Path(destination))


def install(execution, destination):
    if getattr(execution, "_powershard_benchmark", False):
        return False
    from .source_guard import require_signature
    require_signature(getattr(execution, "get_output_data", None), ("prompt_id",), "execution.get_output_data")
    require_signature(getattr(execution.PromptExecutor, "execute_async", None), ("prompt", "prompt_id"), "PromptExecutor.execute_async")
    destination = Path(destination).resolve()
    active = {}
    original_output = execution.get_output_data
    original_prompt = execution.PromptExecutor.execute_async

    @functools.wraps(original_output)
    async def timed_output(prompt_id, unique_id, obj, *args, **kwargs):
        record = active.get(prompt_id)
        if record is None:
            return await original_output(prompt_id, unique_id, obj, *args, **kwargs)
        node_type = record["graph"].get(str(unique_id), {}).get("class_type", getattr(obj, "__name__", type(obj).__name__))
        stage = STAGES.get(node_type, "conditioning_or_other")
        record["inventory"].stage = stage
        before = cuda_snapshot(reset=True)
        start = time.perf_counter()
        result, status = None, "ERROR"
        try:
            result = await original_output(prompt_id, unique_id, obj, *args, **kwargs)
            status = "ASYNC_OR_SUBGRAPH_NOT_TIMED" if result[2] or result[3] else "PASS"
            return result
        finally:
            after = cuda_snapshot()
            record["nodes"].append(dict(node_id=str(unique_id), node_type=node_type, stage=stage,
                                        status=status, wall_s=time.perf_counter()-start,
                                        host_cuda_before=before, host_cuda_after=after))
            record["inventory"].stage = None

    @functools.wraps(original_prompt)
    async def timed_prompt(self, prompt, prompt_id, *args, **kwargs):
        inventory = GPUInventory()
        record = dict(graph=prompt, nodes=[], inventory=inventory)
        active[prompt_id] = record
        inventory.start()
        cuda_snapshot(reset=True)
        start = time.perf_counter()
        status = "ERROR"
        try:
            result = await original_prompt(self, prompt, prompt_id, *args, **kwargs)
            status = "PASS" if getattr(self, "success", False) else "ERROR"
            return result
        finally:
            cuda_snapshot()
            elapsed = time.perf_counter()-start
            inventory.close()
            active.pop(prompt_id, None)
            messages = getattr(self, "status_messages", [])
            cached = [m[1].get("nodes", []) for m in messages if len(m)>1 and m[0]=="execution_cached"]
            from .reporting import redact,environment_identity
            from .runtime import _SESSIONS
            from .topology import process_memory
            from .conditioning_cache import _CACHES
            dump=os.environ.get("POWERSHARD_DEBUG_DUMP_INPUTS")=="1"
            output = dict(schema=2, status=status, run_id=str(prompt_id),prompt_id=prompt_id, graph=redact(prompt,dump),
                          environment=environment_identity(),cpu_memory=process_memory(),
                          conditioning_caches=[c.report() for c in list(_CACHES)],
                          worker_sessions=[dict(role=s.role,fingerprint=getattr(s,"fingerprint",None),
                            report_dir=str(s.report_dir),selected_devices=s.selected_devices,
                            idle_on_cpu=s.idle_on_cpu,last_sequence=s.sequence) for s in list(_SESSIONS)],
                          input_dump_enabled=dump,
                          execution_s=elapsed, start_monotonic_s=start, end_monotonic_s=start+elapsed,
                          host_pid=os.getpid(), queue_time="EXCLUDED", nodes=record["nodes"],
                          cached_nodes=[x for row in cached for x in row],
                          gpu_memory_samples=inventory.samples, gpu_sampling_error=inventory.error,
                          warning_ru="Синхронные native nodes измерены целиком, включая lazy load. Worker load/collectives — отдельные отчёты; не суммировать вложенные времена. Profiler overhead и кэш влияют на сравнение.")
            destination.mkdir(parents=True, exist_ok=True)
            # prompt_id может поступить от внешнего клиента: не использовать как путь.
            target = destination / ("prompt-" + uuid.uuid4().hex + ".json")
            target.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    execution.get_output_data = timed_output
    execution.PromptExecutor.execute_async = timed_prompt
    execution._powershard_benchmark = True
    return True
