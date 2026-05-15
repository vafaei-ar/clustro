"""Experimental VaDE-style latent clustering path.

This implementation approximates a VAE-plus-GMM workflow and should not be
treated as a full mixture-prior ELBO VaDE implementation unless separately
validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clustro.utils.gpu import detect_gpu_status


@dataclass(slots=True)
class VadeArtifacts:
    labels: np.ndarray
    probabilities: np.ndarray
    latent: np.ndarray
    loss: float
    bic: float
    average_confidence: float
    assignment_entropy: float


def fit_predict_vade(
    matrix: np.ndarray,
    params: dict[str, object],
    *,
    seed: int,
    use_gpu_if_available: bool,
    deterministic_mode: str,
) -> VadeArtifacts:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("VaDE requires torch. Install clustro[deep].") from exc

    from sklearn.mixture import GaussianMixture

    latent_dim = int(params.get("latent_dim", params.get("n_clusters", 3)))
    hidden_layers = params.get("hidden_layers", [128, 64])
    if not isinstance(hidden_layers, list):
        hidden_layers = list(hidden_layers)
    n_clusters = int(params.get("n_clusters", 3))
    epochs = int(params.get("epochs", 100))
    batch_size = int(params.get("batch_size", 256))
    learning_rate = float(params.get("learning_rate", 1e-3))

    if seed is not None:
        torch.manual_seed(seed)
        if use_gpu_if_available and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if deterministic_mode == "strict":
        torch.use_deterministic_algorithms(True)

    status = detect_gpu_status(use_gpu_if_available)
    device = torch.device(status.device)
    matrix = np.asarray(matrix, dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(matrix)),
        batch_size=min(batch_size, len(matrix)),
        shuffle=True,
    )

    class _Vae(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder_layers: list[nn.Module] = []
            input_dim = matrix.shape[1]
            for hidden_dim in hidden_layers:
                encoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
                input_dim = hidden_dim
            self.encoder = nn.Sequential(*encoder_layers)
            self.mu = nn.Linear(input_dim, latent_dim)
            self.logvar = nn.Linear(input_dim, latent_dim)

            decoder_layers: list[nn.Module] = []
            input_dim = latent_dim
            for hidden_dim in reversed(hidden_layers):
                decoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
                input_dim = hidden_dim
            decoder_layers.append(nn.Linear(input_dim, matrix.shape[1]))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, inputs):
            hidden = self.encoder(inputs)
            return self.mu(hidden), self.logvar(hidden)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, inputs):
            mu, logvar = self.encode(inputs)
            latent = self.reparameterize(mu, logvar)
            reconstruction = self.decoder(latent)
            return mu, logvar, latent, reconstruction

    model = _Vae().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss(reduction="mean")

    final_loss = 0.0
    for _ in range(epochs):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            mu, logvar, _, reconstruction = model(batch)
            recon_loss = loss_fn(reconstruction, batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        final_loss = total / len(matrix)

    model.eval()
    with torch.no_grad():
        mu, _, _, _ = model(torch.from_numpy(matrix).to(device))
    latent = mu.detach().cpu().numpy()

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=seed)
    labels = gmm.fit_predict(latent)
    probabilities = gmm.predict_proba(latent)
    confidence = float(np.max(probabilities, axis=1).mean())
    entropy = float(-(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=1).mean())
    return VadeArtifacts(
        labels=labels.astype(int),
        probabilities=probabilities.astype(float),
        latent=latent.astype(float),
        loss=float(final_loss),
        bic=float(gmm.bic(latent)),
        average_confidence=confidence,
        assignment_entropy=entropy,
    )
