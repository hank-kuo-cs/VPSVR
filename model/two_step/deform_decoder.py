import torch
import torch.nn as nn


class DeformDecoder(nn.Module):
    def __init__(self, feature_dim: int, vertex_num: int):
        super().__init__()
        self.vertex_num = vertex_num
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + vertex_num * 3, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, vertex_num * 3),
            nn.Tanh()
        )

    def forward(self, vertices: torch.Tensor, global_features: torch.Tensor, local_features: torch.Tensor):
        features = torch.cat([vertices.view(vertices.size(0), -1), global_features, local_features], 1)
        out = self.fc(features)
        deform = out.view(-1, self.vertex_num, 3) * 0.1
        return deform


class DeformGlobalDecoder(nn.Module):
    def __init__(self, feature_dim: int, vertex_num: int, vp_num: int = 16):
        super().__init__()
        self.vertex_num = vertex_num
        self.vp_num = vp_num
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + vertex_num * 3 * 16, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, vertex_num * 3 * vp_num),
            nn.Tanh()
        )

    def forward(self, vertices: torch.Tensor, global_features: torch.Tensor, local_features: torch.Tensor):
        features = torch.cat([vertices.view(vertices.size(0), -1), global_features, local_features], 1)
        out = self.fc(features)
        deform = out.view(-1, self.vertex_num * self.vp_num, 3) * 0.1
        return deform
