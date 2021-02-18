import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from pytorch3d.structures import Meshes
from .cd import ChamferDistanceLoss


cd_loss_func = ChamferDistanceLoss()


class ChamferNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_meshes: list, gt_vertices: list, gt_faces: list):
        self.check_parameters(predict_meshes, gt_vertices, gt_faces)

        B = len(predict_meshes)

        pred_points = torch.cat([m.sample(2048)[0][None] for m in predict_meshes])  # (B, 2048, 3)
        pred_vertices = torch.cat([m.vertices[None] for m in predict_meshes])  # (B, 2048, 3)

        nearest_gt_indices = torch.cat([cd_loss_func(pred_points[b][None], gt_vertices[b][None])[1] for b in range(B)])
        nearest_pred_indices = cd_loss_func(pred_points, pred_vertices)[1]

        nearest_pred_vertices = torch.cat([pred_vertices[i][nearest_pred_indices[i], :][None] for i in range(B)])

        gt_normals = Meshes(gt_vertices, gt_faces).verts_normals_list()  # tuple((N1, 3), (N2, 3), ... (NB, 3))

        nearest_gt_normals = torch.cat([normal[nearest_gt_indices[i], :][None] for i, normal in enumerate(gt_normals)])
        nearest_gt_normals = torch.nn.functional.normalize(nearest_gt_normals, dim=2)

        nearest_pred_edges = pred_points - nearest_pred_vertices
        nearest_pred_edges = torch.nn.functional.normalize(nearest_pred_edges, dim=2)

        normal_loss = torch.abs(torch.sum(nearest_pred_edges * nearest_gt_normals, dim=2)).mean()
        return normal_loss

    @staticmethod
    def check_parameters(predict_meshes: list, gt_vertices: list, gt_faces: list):
        assert isinstance(predict_meshes, list)
        assert isinstance(gt_vertices, list)
        assert isinstance(gt_faces, list)
        assert len(predict_meshes) == len(gt_vertices) == len(gt_faces)
        assert isinstance(predict_meshes[0], TriangleMesh)
