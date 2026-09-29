"""Read-only endpoint существующего ComfyUI server. Нет новых ports/workers."""
import importlib.util
import importlib.metadata
import sys
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
    return tuple(result)


def register_routes():
    # Проверяется наличие server, а не версия. Standalone import остаётся lazy.
    server=sys.modules.get("server")
    instance=getattr(getattr(server,"PromptServer",None),"instance",None)
    if instance is None or getattr(instance,"_powershard_routes",False):return
    from aiohttp import web
    import asyncio
    @instance.routes.get("/powershard/ui_schema")
    async def ui_schema(request):
        from .ui_schema import schema
        return web.json_response(schema())
    @instance.routes.get("/powershard/devices")
    async def inventory(request):
        from .devices import visible_inventory
        try:
            devices=await asyncio.to_thread(visible_inventory)
            return web.json_response(dict(devices=devices,providers=provider_inventory()))
        except (OSError,RuntimeError,ValueError) as error:
            return web.json_response(dict(devices=[],error=str(error),providers=provider_inventory()))
    instance._powershard_routes=True
