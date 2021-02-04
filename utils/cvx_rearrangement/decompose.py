import torch
import trimesh
from ..loss import ChamferDistanceLoss


def get_part_meshes(mesh, class_id) -> (list, list):
    convex_hull_num = 16 if class_id == '03001627' else 8
    convex_hulls = mesh.convex_decomposition(convex_hull_num, concavity=0.01)
    if not isinstance(convex_hulls, list):
        convex_hulls = [convex_hulls]

    convex_types = torch.full((len(convex_hulls),), fill_value=-2,
                              dtype=torch.int)  # -2: none, -1: center, >= 0: symmetric with number it save

    centers = []
    symmetries = []
    skip_indices = []

    for i in range(len(convex_hulls)):
        if is_mesh_center(convex_hulls[i]):
            centers.append(i)
            convex_types[i] = -1
            continue

        if i in skip_indices:
            continue

        for j in range(i + 1, len(convex_hulls)):
            if is_two_mesh_symmetric(convex_hulls[i], convex_hulls[j]):
                if j in skip_indices:
                    continue

                symmetries.append((i, j))
                convex_types[i] = j
                convex_types[j] = i

                skip_indices.append(j)
                break

    part_meshes = []
    skip_indices = []
    symmetry_indices = []  # 0: none, 1: center, 2: symmetry,

    for i, convex_type in enumerate(convex_types):
        if i in skip_indices:
            continue

        if convex_type == -2:
            part_meshes.append(convex_hulls[i])
            symmetry_indices.append(0)

        elif convex_type == -1:
            part_meshes.append(convex_hulls[i])
            symmetry_indices.append(1)

        else:
            part_meshes.append(trimesh.boolean.union([convex_hulls[i], convex_hulls[convex_type.item()]]))
            symmetry_indices.append(2)
            skip_indices.append(convex_type.item())

    return part_meshes, symmetry_indices


def is_two_mesh_symmetric(mesh1, mesh2):
    cd_loss_func = ChamferDistanceLoss()

    sample_points1 = torch.tensor(mesh1.sample(256))[None]
    sample_points2 = torch.tensor(mesh2.sample(256))[None]

    volume1, volume2 = get_volume(mesh1), get_volume(mesh2)
    is_volume_similar = bool(max([volume1, volume2]) / min([volume1, volume2]) < 4)

    center1, center2 = sample_points1[0].mean(0), sample_points2[0].mean(0)
    center_dist = get_dist_points(map_from_xy_plane(center1), center2)
    is_center_close = bool(center_dist <= 0.15 or center_dist <= max([volume1, volume2]) * 2)

    cd_loss = cd_loss_func(map_from_xy_plane(sample_points1), sample_points2)[0]
    is_cd_low = bool(cd_loss <= max([volume1, volume2]) / 4 or cd_loss <= 0.15)

    return is_center_close and is_cd_low and is_volume_similar


def is_mesh_center(mesh: trimesh.Trimesh):
    center = get_center_point(mesh)
    z_range = mesh.extents[2]
    volume = get_volume(mesh)

    return abs(center[2]) < 0.03 or abs(center[2]) < (z_range / 10) or abs(center[2]) < (z_range / (4 * (1 / volume)))


def get_center_point(mesh: trimesh.Trimesh):
    vertices = torch.tensor(mesh.vertices)
    return vertices.mean(0)


def get_volume(mesh: trimesh.Trimesh):
    v1, v2, v3 = mesh.extents
    return v1 * v2 * v3


def map_from_xy_plane(points):
    points[..., 2] *= -1
    return points


def get_dist_points(points1, points2):
    return torch.norm(points1 - points2)
