import torch
import torch.nn as nn


class DeformDecoder(nn.Module):
    def __init__(self, feature_dim: int, vertex_num: int):
        super().__init__()
        self.vertex_num = vertex_num
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, vertex_num * 3),
            nn.Tanh()
        )

    def forward(self, global_features, local_features):
        features = torch.cat([global_features, local_features], 1)
        out = self.fc(features)
        deform = out.view(-1, self.vertex_num, 3) * 0.1
        return deform
