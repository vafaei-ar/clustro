"""DEC clustering implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clustro.repr.ae_repr import train_autoencoder
from clustro.utils.gpu import detect_gpu_status


@dataclass(slots=True)
class DecArtifacts:
    labels: np.ndarray
    loss: float
    latent: np.ndarray
    reconstruction_loss: float
    average_confidence: float
    assignment_entropy: float
    iterations: int


def fit_predict_dec(
    matrix: np.ndarray,
    params: dict[str, object],
    *,
    seed: int,
    use_gpu_if_available: bool,
    deterministic_mode: str,
) -> DecArtifacts:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("DEC requires torch. Install clustro[deep].") from exc

    from sklearn.cluster import KMeans

    latent_dim = int(params.get("latent_dim", params.get("n_clusters", 3)))
    n_clusters = int(params.get("n_clusters", 3))
    hidden_layers = params.get("hidden_layers", [128, 64])
    if not isinstance(hidden_layers, list):
        hidden_layers = list(hidden_layers)
    autoencoder = train_autoencoder(
        matrix,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        dropout=float(params.get("dropout", 0.0)),
        epochs=int(params.get("pretrain_epochs", 75)),
        batch_size=int(params.get("batch_size", 256)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        early_stopping_patience=int(params.get("early_stopping_patience", 10)),
        random_state=seed,
        use_gpu_if_available=use_gpu_if_available,
        deterministic_mode=deterministic_mode,
    )

    latent = autoencoder.latent.astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(latent)
    centers = kmeans.cluster_centers_.astype(np.float32)

    status = detect_gpu_status(use_gpu_if_available)
    device = torch.device(status.device)
    z = torch.tensor(latent, device=device)
    cluster_centers = torch.nn.Parameter(torch.tensor(centers, device=device))
    optimizer = torch.optim.Adam([cluster_centers], lr=float(params.get("finetune_learning_rate", 1e-3)))
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    tol = float(params.get("tolerance", 1e-4))
    patience = int(params.get("finetune_patience", 5))

    final_loss = 0.0
    best_loss = float("inf")
    stale_epochs = 0
    epochs_run = 0
    for _ in range(int(params.get("finetune_epochs", 50))):
        q = _soft_assign(z, cluster_centers)
        p = _target_distribution(q)
        loss = loss_fn(torch.log(torch.clamp(q, 1e-8, 1.0)), p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        epochs_run += 1
        if best_loss - final_loss > tol:
            best_loss = final_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    with torch.no_grad():
        q = _soft_assign(z, cluster_centers)
        labels = torch.argmax(q, dim=1).cpu().numpy().astype(int)
        confidence = torch.max(q, dim=1).values.mean().item()
        entropy = (-(q * torch.log(torch.clamp(q, 1e-8, 1.0))).sum(dim=1)).mean().item()
    return DecArtifacts(
        labels=labels,
        loss=final_loss,
        latent=latent.astype(float),
        reconstruction_loss=autoencoder.reconstruction_loss,
        average_confidence=float(confidence),
        assignment_entropy=float(entropy),
        iterations=epochs_run,
    )


def _soft_assign(z, centers):
    import torch

    distances = torch.sum((z.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
    numerator = 1.0 / (1.0 + distances)
    numerator = numerator ** ((1 + 1.0) / 2.0)
    return numerator / numerator.sum(dim=1, keepdim=True)


def _target_distribution(q):
    weight = q**2 / q.sum(dim=0, keepdim=True)
    return weight / weight.sum(dim=1, keepdim=True)
