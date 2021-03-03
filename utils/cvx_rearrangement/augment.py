import torch
import random
import trimesh
from kaolin.rep import TriangleMesh
from ..transform import rotate_points
from .mesh import meshes_trimesh2kaolin


def augment_mesh(part_meshes: list, symmetry_indices: list, class_id: str) -> TriangleMesh:
    cutout_meshes = random_cutout(part_meshes, symmetry_indices)
    cutout_mesh = trimesh.boolean.union(cutout_meshes)
    cutout_kmesh = meshes_trimesh2kaolin([cutout_mesh])[0]
    trans_kmesh = random_xy_translate(cutout_kmesh)
    scale_kmesh = random_scale(trans_kmesh)
    augment_kmesh = scale_kmesh
    if class_id != '03001627':
        rotate_kmesh = random_z_rotate(scale_kmesh)
        augment_kmesh = rotate_kmesh

    return augment_kmesh


def rotate_points_forward_vec(points: torch.Tensor, angle: float, vec: list):
    assert points.ndimension() == 2

    device = points.device

    if not points.is_cuda:
        points = points.cuda()

    v = torch.tensor([vec], dtype=torch.float).cuda()

    angles = torch.tensor([[angle]], dtype=torch.float).cuda() / 360
    q = torch.cat([v, angles], dim=1).cuda()

    points = rotate_points(points[None], q)

    return points[0].detach().to(device)


def random_cutout(part_meshes: list, symmetry_indices: list) -> list:
    assert isinstance(part_meshes[0], trimesh.Trimesh)

    symmetry_indices = [idx for idx, sym_type in enumerate(symmetry_indices) if sym_type > 0]
    if len(symmetry_indices) < 2:
        return part_meshes

    cutout_num = random.randint(0, len(symmetry_indices) - 1)
    cutout_indices = random.sample(symmetry_indices, cutout_num)

    return [part_mesh for idx, part_mesh in enumerate(part_meshes) if idx not in cutout_indices]


def random_scale(mesh: TriangleMesh) -> TriangleMesh:
    scales = (torch.rand(3) + 0.5).to(mesh.vertices.device)  # [0.5, 1.5]

    mesh.vertices *= scales
    return mesh


def random_xy_translate(mesh: TriangleMesh) -> TriangleMesh:
    x_translate = (torch.rand(1).item() - 0.5) / 2  # [-0.25, 0.25]
    y_translate = (torch.rand(1).item() - 0.5) / 4  # [-0.125, 0.125]

    mesh.vertices[..., 0] += x_translate
    mesh.vertices[..., 1] += y_translate
    return mesh


def random_z_rotate(mesh: TriangleMesh) -> TriangleMesh:
    z_axis = [0, 0, 1]
    angle = random.choice([90.0, -90.0, 0.0, 180.0, -180.0])

    mesh.vertices = rotate_points_forward_vec(mesh.vertices.float(), angle, z_axis)
    return mesh

