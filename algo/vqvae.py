import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ---------------------- VQ-VAE Components ----------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # inputs: (batch, num_codes, embedding_dim)
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)  # (B * N, D)

        distances = (
            flat_input.pow(2).sum(1, keepdim=True)
            - 2 * flat_input @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )  # (B * N, K)

        encoding_indices = torch.argmin(distances, dim=1)  # (B * N)
        quantized = self.embedding(encoding_indices).view(*input_shape)  # (B, N, D)

        # VQ losses
        codebook_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = F.mse_loss(quantized, inputs.detach())

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, encoding_indices.view(input_shape[0], input_shape[1]), codebook_loss, commitment_loss


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        code_dim: int = 32,
        num_codes: int = 8,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_codes = num_codes
        hidden_dim = 512

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_codes * code_dim),
        )
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=code_dim, commitment_cost=commitment_cost)
        self.decoder = nn.Sequential(
            nn.Linear(num_codes * code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, input_dim)
        z_e = self.encoder(x).view(x.shape[0], self.num_codes, self.code_dim)  # (B, N, D)
        z_q, indices, codebook_loss, commitment_loss = self.quantizer(z_e)
        z_q_flat = z_q.reshape(x.shape[0], -1)
        x_recon = self.decoder(z_q_flat)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, z_q_flat, recon_loss, codebook_loss, commitment_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x).view(x.shape[0], self.num_codes, self.code_dim)
        z_q, _, _, _ = self.quantizer(z_e)
        return z_q.reshape(x.shape[0], -1)