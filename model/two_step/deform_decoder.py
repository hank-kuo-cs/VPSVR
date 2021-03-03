import torch
import torch.nn as nn


class DeformDecoder(nn.Module):
    def __init__(self, feature_dim=960+512, vertex_num=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, vertex_num * 3),
        )
        self.tanh = nn.Tanh()

    def forward(self, global_features, local_features):
        features = torch.cat([global_features, local_features], 1)
        out = self.fc(features)
        deform = self.tanh(out) * 0.2
        deform = deform.view(-1, 128, 3)
        return deform
