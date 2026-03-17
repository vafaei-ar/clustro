"""Optional Ray bootstrap helper."""

from __future__ import annotations


def maybe_init_ray(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import ray  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Ray requested but not installed. Use clustro[tracking].") from exc
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
