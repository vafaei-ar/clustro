"""GPU and accelerator detection helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GpuStatus:
    requested: bool
    torch_available: bool
    cuda_available: bool
    rapids_available: bool
    device: str


def detect_gpu_status(requested: bool) -> GpuStatus:
    torch_available = False
    cuda_available = False
    try:
        import torch

        torch_available = True
        if requested:
            cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_available = False
        cuda_available = False

    try:
        import cuml  # type: ignore  # noqa: F401

        rapids_available = True
    except ImportError:
        rapids_available = False

    device = "cuda" if requested and cuda_available else "cpu"
    return GpuStatus(
        requested=requested,
        torch_available=torch_available,
        cuda_available=cuda_available,
        rapids_available=rapids_available,
        device=device,
    )
