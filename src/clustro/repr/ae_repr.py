"""Autoencoder-based latent representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from clustro.repr.base import RepresentationResult
from clustro.utils.gpu import detect_gpu_status


@dataclass(slots=True)
class AutoencoderArtifacts:
    latent: np.ndarray
    reconstruction_loss: float
    metadata: dict[str, Any]


class AutoencoderRepresentation:
    name = "autoencoder"

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_layers: list[int] | tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        random_state: int | None = None,
        use_gpu_if_available: bool = False,
        deterministic_mode: str = "fast",
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_layers = list(hidden_layers)
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.use_gpu_if_available = use_gpu_if_available
        self.deterministic_mode = deterministic_mode

    def fit_transform(self, matrix: np.ndarray) -> RepresentationResult:
        artifacts = train_autoencoder(
            matrix,
            latent_dim=self.latent_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            early_stopping_patience=self.early_stopping_patience,
            random_state=self.random_state,
            use_gpu_if_available=self.use_gpu_if_available,
            deterministic_mode=self.deterministic_mode,
        )
        return RepresentationResult(
            matrix=artifacts.latent,
            metadata={
                "name": self.name,
                "reconstruction_loss": artifacts.reconstruction_loss,
                **artifacts.metadata,
            },
        )


def train_autoencoder(
    matrix: np.ndarray,
    *,
    latent_dim: int,
    hidden_layers: list[int] | tuple[int, ...],
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    random_state: int | None,
    use_gpu_if_available: bool,
    deterministic_mode: str,
) -> AutoencoderArtifacts:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Autoencoder representation requires torch. Install clustro[deep]."
        ) from exc

    if random_state is not None:
        torch.manual_seed(random_state)
        if use_gpu_if_available and torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    if deterministic_mode == "strict":
        torch.use_deterministic_algorithms(True)
        if use_gpu_if_available and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    status = detect_gpu_status(use_gpu_if_available)
    device = torch.device(status.device)
    matrix = np.asarray(matrix, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(matrix))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(matrix)), shuffle=True)

    class _Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder_layers: list[nn.Module] = []
            input_dim = matrix.shape[1]
            for hidden_dim in hidden_layers:
                encoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
                if dropout > 0:
                    encoder_layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            encoder_layers.append(nn.Linear(input_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers: list[nn.Module] = []
            input_dim = latent_dim
            for hidden_dim in reversed(hidden_layers):
                decoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
                input_dim = hidden_dim
            decoder_layers.append(nn.Linear(input_dim, matrix.shape[1]))
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, inputs):
            latent = self.encoder(inputs)
            reconstruction = self.decoder(latent)
            return latent, reconstruction

    model = _Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch)
        epoch_loss /= len(matrix)
        if epoch_loss + 1e-8 < best_loss:
            best_loss = epoch_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        latent, _ = model(torch.from_numpy(matrix).to(device))
    latent_np = latent.detach().cpu().numpy().astype(float)
    return AutoencoderArtifacts(
        latent=latent_np,
        reconstruction_loss=float(best_loss),
        metadata={
            "latent_dim": latent_dim,
            "device": status.device,
            "torch_cuda_available": status.cuda_available,
        },
    )
