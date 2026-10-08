"""Diagnostics and pure JSON migration on the existing ComfyUI server."""
import importlib.util
import importlib.metadata
import sys
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _package_map(directory_stamp):
    return importlib.metadata.packages_distributions()


def _distributions():
    # Добавление/удаление dist-info меняет mtime каталога. Сами binaries всё ещё
    # stat'ятся при каждом fingerprint, без повторного разбора всех METADATA.
    stamp=[]
    for value in sys.path:
        path=Path(value or ".")
        if path.is_dir():
            stamp.append((str(path.resolve()),path.stat().st_mtime_ns))
    return _package_map(tuple(stamp))


@lru_cache(maxsize=32)
def _binary_members(distname, version, directory_stamp):
    distribution=importlib.metadata.distribution(distname)
    return tuple(Path(distribution.locate_file(m)) for m in distribution.files or () if str(m).endswith((".so",".pyd")))


def provider_inventory():
    rows=[]
    distributions=_distributions()
    for name in ("torch","flash_attn","vllm_flash_attn","sageattention"):
        spec=importlib.util.find_spec(name)
        path=str(spec.origin) if spec else None
        loaded=sys.modules.get(name)
        version=getattr(loaded,"__version__",None)
        rows.append(dict(name=name,path=path,
            distributions={d:importlib.metadata.version(d) for d in distributions.get(name,[])},
            status="NOT_PROBED" if spec else "NOT_INSTALLED",
            legacy_shim=bool(path and "flash_attn_shim" in path or "shim" in str(version)),
            reason="Наличие модуля не доказывает работоспособность CUDA kernel" if spec else "Опциональный пакет не найден"))
    return rows


def provider_stamp():
    # Не импортирует optional modules, не запускает kernels.
    result=[]
    distributions=_distributions()
    for name in ("torch","flash_attn","vllm_flash_attn","sageattention"):
        spec=importlib.util.find_spec(name)
        path=Path(spec.origin) if spec and spec.origin else None
        stat=path.stat() if path and path.is_file() else None
        binaries=[]
        for distname in distributions.get(name,[]):
            distribution=importlib.metadata.distribution(distname)
            base=Path(distribution.locate_file(""))
            for binary in _binary_members(distname,distribution.version,base.stat().st_mtime_ns):
                if binary.is_file():
                    info=binary.stat();binaries.append((str(binary),info.st_size,info.st_mtime_ns))
        result.append((name,str(path),stat.st_mtime_ns if stat else None,stat.st_size if stat else None,tuple(sorted(binaries))))
    custom=Path(os.environ.get("POWERSHARD_FLASH_ATTN_SHIM",str(Path(__file__).with_name("flash_attn_shim.py"))))
    stat=custom.stat() if custom.is_file() else None
    result.append(("scoped_flash_shim",str(custom),stat.st_mtime_ns if stat else None,stat.st_size if stat else None,()))
    return tuple(result)


def runtime_status():
    from .runtime import _SESSIONS
    import json
    result=[]
    for session in list(_SESSIONS):
        metrics=[];progress=[];idle=[]
        for event in reversed(session.history):
            parked=[r.get("phase_offload") for r in event.get("ranks",[]) if r.get("phase_offload")]
            if parked and not idle:idle=parked
            rows=[r.get("metrics") for r in event.get("ranks",[]) if r.get("metrics")]
            if rows:
                metrics=rows
                break
        if session.path is not None:
            for rank in range(len(session.processes)):
                path=session.report_dir/f"{session.path.name}-rank{rank}-progress.json"
                try:progress.append(json.loads(path.read_text()))
                except (OSError,ValueError):pass
        result.append(dict(role=session.role,running=session.running,draining=session.draining,
            retained=session.retains_weights,idle_on_cpu=session.idle_on_cpu,
            requested=session.config.requested_attention,
            selected=(session.attention_policy or {}).get("effective"),
            placement=session.config.weight_placement,
            idle_placement="cpu" if session.idle_on_cpu else None,
            phase_offload=idle if session.idle_on_cpu else [],
            gpu_ids=list(session.config.gpu_ids),
            ranks=[dict(rank=i,memory=idle[i].get("after") if session.idle_on_cpu and i<len(idle) else m.get("memory"),attention=m.get("attention",{}).get("effective_used"),
                        provider=m.get("attention",{}).get("provider"),
                        weights=dict(cpu_shard_bytes=idle[i].get("cpu_shard_bytes",0),placement="cpu",gpu_shard_bytes=0,managed_shard_bytes=0)
                            if session.idle_on_cpu and i<len(idle) else m.get("weight_memory",dict(cpu_shard_bytes=m.get("cpu_shard_bytes",0),
                                    persistent_shard_bytes=m.get("persistent_shard_bytes",0))),
                        prefetch=m.get("memory_plan",{}),wall_phases=m.get("wall_phases",{}),
                        mlp_chunks=sorted(set(v.get("effective_tokens",0) for v in m.get("mlp",{}).values())))
                   for i,m in enumerate(metrics)], progress=progress))
    return result


def register_routes():
    # Проверяется наличие server, а не версия. Standalone import остаётся lazy.
    server=sys.modules.get("server")
    instance=getattr(getattr(server,"PromptServer",None),"instance",None)
    if instance is None or getattr(instance,"_powershard_routes",False):return
    from aiohttp import web
    import asyncio
    @instance.routes.get("/powershard/devices")
    async def inventory(request):
        from .devices import visible_inventory
        try:
            devices=await asyncio.to_thread(visible_inventory)
            return web.json_response(dict(devices=devices,providers=provider_inventory()))
        except (OSError,RuntimeError,ValueError) as error:
            return web.json_response(dict(devices=[],error=str(error),providers=provider_inventory()))
    @instance.routes.post("/powershard/migrate-workflow")
    async def migrate_workflow(request):
        from .workflow_migration import migrate_ui
        try:
            graph, changes = migrate_ui(await request.json())
            return web.json_response(dict(graph=graph, changes=changes))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            return web.json_response(dict(error=str(error)), status=400)
    @instance.routes.get("/powershard/status")
    async def status(request):
        return web.json_response(dict(sessions=runtime_status(),
            note="Последний завершённый RPC и текущий progress; installed/selected не доказывают фактический CUDA kernel. Triton не вызывается PowerShard."))
    instance._powershard_routes=True
