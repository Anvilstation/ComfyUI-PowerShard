"""Только host sampler boundary и JSON metadata; module patch исполняется в workers."""
import time
import uuid
import warnings
import torch
from .conditioning_cache import content_hash


def sampler_capability(sampler, config=None):
    from comfy.k_diffusion.sampling import sample_euler
    allow_any = bool(getattr(config, "allow_any_sampler", False))
    if getattr(sampler,"sampler_function",None) is not sample_euler:
        if allow_any:
            # Opt-in: mechanism forecast/capture семантически не зависит от
            # sampler — capture берёт последний block hidden, predict
            # подменяет target до native heads. Смена траектории и так
            # присуща Spectrum (approximate). Смешанные внутренние вызовы
            # мультистейдж-солверов могут дать деградацию: это на совести
            # включившего, метрики actual/forecast в отчёте покажут распределение.
            return True,"allow_any_sampler: нативный deterministic Euler не гарантирован, прогноз включён"
        return False,"Spectrum bridge проверен только для native deterministic Euler; выполняется обычный FSDP"
    if getattr(sampler,"extra_options",{}).get("s_churn",0)!=0:
        return False,"Spectrum: stochastic Euler s_churn != 0; выполняется обычный FSDP"
    return True,"native_euler"


def spectrum_outer_sample(executor,noise,latent_image,sampler,sigmas,*args,**kwargs):
    from .runtime import _PHASE_LOCK,wait_for_pending_phase
    import comfy.model_management
    wait_for_pending_phase(comfy.model_management.throw_exception_if_processing_interrupted)
    # Одна GPU sampling phase. Не позволять двум clones одной session заменить
    # run context между forward, если внешний executor использует несколько threads.
    with _PHASE_LOCK:
        return _run_sampling(executor,noise,latent_image,sampler,sigmas,*args,**kwargs)


def _run_sampling(executor,noise,latent_image,sampler,sigmas,*args,**kwargs):
    patcher=executor.class_obj.model_patcher
    session=patcher.session
    from .spectrum_config import SpectrumConfig
    spectrum_policy=SpectrumConfig(**session.role_options.get("spectrum",{}))
    eligible,reason=sampler_capability(sampler,spectrum_policy)
    enabled=session.role_options.get("spectrum",{}).get("enabled",False)
    if enabled and not eligible:warnings.warn(reason)
    sampling=patcher.get_model_object("model_sampling")
    timesteps=sampling.timestep(sigmas.detach()).float().cpu().flatten().tolist()
    run_id=uuid.uuid4().hex
    from .reporting import describe_options
    tensors=lambda x:[x] if isinstance(x,torch.Tensor) else list(x.unbind())
    manifest=dict(seed=kwargs.get("seed",args[3] if len(args)>3 else None),noise_sha256=content_hash(tensors(noise)),
        latent_shapes=[list(t.shape) for t in tensors(latent_image)] if latent_image is not None else None,
        sampler=getattr(getattr(sampler,"sampler_function",None),"__name__",type(sampler).__name__),
        sampler_options=describe_options(getattr(sampler,"extra_options",{})),
        checkpoint_revision="See worker checkpoint identity; remote revision UNKNOWN unless catalog match",
        lora="UNSUPPORTED in supplied baseline; no descriptors silently applied",
        cfg=getattr(executor.class_obj,"cfg",None))
    session.sampling_context=dict(run_id=run_id,eligible=eligible,reason=reason,
        timesteps=timesteps,sigmas=sigmas.detach().float().cpu().flatten().tolist(),
        steps=max(0,len(timesteps)-1),started=time.perf_counter(),history_start=len(session.history),manifest=manifest)
    failed=False
    try:
        return executor(noise,latent_image,sampler,sigmas,*args,**kwargs)
    except BaseException:
        failed=True
        raise
    finally:
        context=session.sampling_context
        if context is not None:context["status"]="INTERRUPTED_OR_FAILED" if failed else "COMPLETE"
        session.sampling_context=None
        cleanup_error=None
        try:
            session.finish_sampling()
        except BaseException as error:
            cleanup_error=error
        try:
            if context is not None:session.save_run_summary(context)
        except BaseException as error:
            cleanup_error=cleanup_error or error
        if cleanup_error is not None:
            if failed:warnings.warn(f"PowerShard sampler cleanup failed: {cleanup_error}; original sampling error preserved")
            else:raise cleanup_error


def forward_metadata(session,x,timestep,context,kwargs,options):
    state=getattr(session,"sampling_context",None)
    if not state:return dict(eligible=False,reason="missing_native_sampler_context")
    out={k:state[k] for k in ("run_id","eligible","reason","steps")}
    if not out["eligible"]:return out
    values=timestep.detach().float().cpu().flatten().tolist()
    if not values or any(v!=values[0] for v in values):
        return dict(out,eligible=False,reason="mixed_timesteps_in_batch")
    schedule=state["timesteps"][:-1]
    if not schedule:return dict(out,eligible=False,reason="empty_schedule")
    index=min(range(len(schedule)),key=lambda i:abs(schedule[i]-values[0]))
    if abs(schedule[index]-values[0])>max(1e-5,abs(values[0])*1e-5):
        return dict(out,eligible=False,reason="off_schedule_evaluation")
    lo,hi=min(state["timesteps"]),max(state["timesteps"])
    if hi<=lo:return dict(out,eligible=False,reason="constant_schedule")
    def clean(v):
        if isinstance(v,uuid.UUID):return str(v)
        if type(v).__name__=="PackedLayout" and type(v).__module__=="comfy.ldm.minimax.model":return None
        if isinstance(v,dict):return {k:clean(a) for k,a in v.items() if k not in ("wrappers","layout")}
        if isinstance(v,(list,tuple)):return [clean(a) for a in v]
        return v
    # Не хешируем noisy latents как conditioning: они меняются каждый diffusion step.
    # Context/reference/masks включены по содержимому. Hash CPU cost виден в run wall time.
    stable_options={k:v for k,v in options.items() if k not in ("sigmas","block_index")}
    key=content_hash(dict(context=context,conditioning=clean(kwargs),options=clean(stable_options),
                          latent_shapes=[list(t.shape) for t in x]))
    return dict(out,step_index=index,coordinate=2*(values[0]-lo)/(hi-lo)-1,conditioning_key=key)
