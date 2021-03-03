import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from torch_geometric.nn import GCNConv
from .perceptual_feature import get_local_features


class DeformGCN(nn.Module):
    def __init__(self, n_dim=3, feature_dim=960 + 512, v_num=2048):
        super().__init__()
        conv = GCNConv
        self.relu = nn.LeakyReLU()

        self.encoder = nn.ModuleList([
            conv(n_dim + feature_dim, 512),
            conv(512, 512),
            conv(512, 256),
            conv(256, 256),
            conv(256, 64),
            conv(64, 3),
        ])

        self.decoder = nn.Sequential(
            nn.Linear(v_num * 3, v_num * 3),
            nn.Tanh()
        )

    def forward(self, meshes: list, imgs: torch.Tensor, perceptual_features: list, global_features: torch.Tensor):
        B = len(meshes)
        N = len(meshes[0].vertices)

        batch_vertices = self.get_batch_vertices(meshes)  # (B, N, 3)
        edges = self.get_edges(meshes[0])  # (2, K)

        global_features = global_features[:, None, :].repeat(1, N, 1)
        local_features = get_local_features(batch_vertices, imgs, perceptual_features)

        x = torch.cat([batch_vertices, local_features, global_features], 2)

        features = self.extract_features(x, edges).view(B, -1)

        deformations = self.decoder(features).view(B, -1, 3) * 0.1

        return deformations

    @staticmethod
    def get_edges(mesh: TriangleMesh):
        vertices, faces = mesh.vertices, mesh.faces

        face_size= faces.shape[1]
        edges = torch.cat([faces[:, i:i + 2] for i in range(face_size - 1)] +
                          [faces[:, [-1, 0]]], dim=0)

        edges = torch.sort(edges, dim=1)[0]
        edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
        edges = torch.cat([edges, edges.flip(1)]).view(2, -1)

        return edges

    @staticmethod
    def get_batch_vertices(meshes: list):
        return torch.cat([mesh.vertices[None] for mesh in meshes])

    def extract_features(self, x, edges):
        for i, conv in enumerate(self.encoder):
            x = conv(x, edges, None)
            if i % 2:
                x = self.relu(x)
        return x
