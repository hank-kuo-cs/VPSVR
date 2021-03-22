import torch
import torch.nn as nn


sigmoid = nn.Sigmoid()


class VolumeRotateDecoder(nn.Module):
    def __init__(self, feature_dim: int, volume_eps=0.1, volume_restrict=None):
        super().__init__()
        self.volume_eps = volume_eps
        self.feature_dim = feature_dim
        self.volume_restrict = volume_restrict if volume_restrict is not None else [8, 10, 12]

        self.fc = self._make_linear(self.feature_dim)

    def forward(self, local_features: torch.Tensor):
        out = self.fc(local_features)

        volumes = out[..., :3]
        rotates = out[..., 3:]

        volumes = sigmoid(volumes) + self.volume_eps
        rotates = sigmoid(rotates)

        for i in range(3):
            volumes[..., i] = torch.div(volumes[..., i], self.volume_restrict[i])

        return volumes, rotates

    @staticmethod
    def _make_linear(input_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 7),
        )
