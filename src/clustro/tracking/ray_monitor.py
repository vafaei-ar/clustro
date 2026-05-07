"""Optional Ray bootstrap helper."""

from __future__ import annotations


def maybe_init_ray(enabled: bool, *, n_jobs: int | None = None) -> bool:
    if not enabled:
        return False
    try:
        import ray  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Ray requested but not installed. Use clustro[tracking].") from exc
    if not ray.is_initialized():
        init_kwargs = {"ignore_reinit_error": True}
        if n_jobs is not None and n_jobs > 0:
            init_kwargs["num_cpus"] = n_jobs
        ray.init(**init_kwargs)
    return True
