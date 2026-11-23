import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Encoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        use_rnn: bool = True,
        rnn_num_layers: int = 1,
        hidden_dim: int = 256,
        use_norm: bool = True,
    ):
        super().__init__()
        self.use_rnn = use_rnn
        self.rnn_num_layers = rnn_num_layers
        self.hidden_dim = hidden_dim
        self.use_norm = use_norm

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if use_rnn:
            self.gru = nn.GRU(hidden_dim, hidden_dim, rnn_num_layers)
        else:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim) if self.use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if self.use_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(hidden_dim) if self.use_norm else nn.Identity()

    def forward(self, state: torch.Tensor, rnn_state: torch.Tensor = None):
        feat = self.fc1(state)
        feat = self.norm1(feat)
        feat = F.relu(feat)
        feat = self.fc2(feat)
        feat = self.norm2(feat)
        feat = F.relu(feat)
        if self.use_rnn:
            feat = feat.unsqueeze(0)
            out, rnn_state = self.gru(feat, rnn_state.transpose(0, 1))
            rnn_state = rnn_state.transpose(0, 1)
            feat = out.squeeze(0)
            feat = self.norm3(feat)
        else:
            feat = self.fc3(feat)
            feat = self.norm3(feat)
            feat = F.relu(feat)
        return feat, rnn_state


class PowerPredictor(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        use_norm: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_norm = use_norm
        self.hidden_dim = hidden_dim

        input_dim = state_dim + action_dim
        if encoder is not None:
            input_dim = encoder.hidden_dim + action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.norm1 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()

        self.apply(weight_init)

    def forward(self, x, a, rnn_state=None):
        feat, _ = self.encode(x, rnn_state)
        feat = torch.cat([feat, a], dim=1)
        p = F.relu(self.norm1(self.fc1(feat)))
        p = F.relu(self.norm2(self.fc2(p)))
        p = self.fc3(p)

        return p.squeeze(1)

    def encode(self, x, rnn_state=None):
        if self.encoder is not None:
            feat, rnn_state = self.encoder(x, rnn_state)
        else:
            feat = x
        return feat, rnn_state
