import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from ..perceptual_feature import get_local_features
from .bottleneck import GBottleneck, GCNConv


class DeformGCN(nn.Module):
    def __init__(self, n_dim=3, feature_dim=960 + 128, v_num=2048):
        super().__init__()
        self.v_num = v_num
        self.edges = None
        self.gcn = nn.ModuleList([
            GBottleneck(in_dim=n_dim + feature_dim, hidden_dim=256, out_dim=64),
            GCNConv(64, 3)
        ])
        self.global_split_fc = nn.Linear(512, 128 * v_num)

    def forward(self, meshes: list, imgs: torch.Tensor, perceptual_features: list, global_features: torch.Tensor):
        B = len(meshes)

        batch_vertices = self.get_batch_vertices(meshes)  # (B, N, 3)

        if self.edges is None:
            self.edges = self.get_edges(meshes[0])

        global_features = self.global_split_fc(global_features).view(B, self.v_num, 128)
        local_features = get_local_features(batch_vertices, imgs, perceptual_features)

        x = torch.cat([batch_vertices, global_features, local_features], 2)  # (B, N, 3+128+960)

        for conv in self.gcn:
            x = conv(x, self.edges)

        return x  # (B, N, 3)

    @staticmethod
    def get_edges(mesh: TriangleMesh):
        vertices, faces = mesh.vertices, mesh.faces

        face_size = faces.shape[1]
        edges = torch.cat([faces[:, i:i + 2] for i in range(face_size - 1)] +
                          [faces[:, [-1, 0]]], dim=0)

        edges = torch.sort(edges, dim=1)[0]
        edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
        edges = torch.cat([edges, edges.flip(1)]).view(2, -1)

        return edges

    @staticmethod
    def get_batch_vertices(meshes: list):
        return torch.cat([mesh.vertices[None] for mesh in meshes])
