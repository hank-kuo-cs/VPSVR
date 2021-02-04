import torch
import trimesh
from kaolin.rep import TriangleMesh


def merge_meshes(meshes: list, is_kaolin=False):
    assert len(meshes) > 0 and isinstance(meshes[0], TriangleMesh)

    vertices = []
    faces = []

    vertices_num = 0

    for mesh in meshes:
        vertices_now = mesh.vertices.cpu().detach().float()
        faces_now = mesh.faces.cpu().detach().long()

        vertices.append(vertices_now)
        faces.append(faces_now + vertices_num)

        vertices_num += len(vertices_now)

    return trimesh.Trimesh(torch.cat(vertices), torch.cat(faces)) if not is_kaolin \
        else TriangleMesh.from_tensors(torch.cat(vertices), torch.cat(faces))


def merge_parts_and_get_colors(part_meshes: list, device='cuda'):
    vertex_num = 0
    vertices, faces, texture, uv = [], [], [], []

    for i, mesh in enumerate(part_meshes):
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + vertex_num)
        uv.append(torch.full(size=(mesh.vertices.size(0), 2), fill_value=i/len(part_meshes) + 0.01))
        texture.append(get_random_colors())

        vertex_num += mesh.vertices.size(0)

    vertices = torch.cat(vertices)
    faces = torch.cat(faces)

    uv = torch.cat(uv)[None].to(device)
    texture = torch.cat(texture, 2)[None].to(device)

    merged_mesh = TriangleMesh.from_tensors(vertices, faces)
    merged_mesh.to(device)

    return merged_mesh, uv, texture


def get_random_colors():
    c = torch.rand(3)
    return torch.cat([torch.full((1, 1, 1), c[i].item()) for i in range(3)], 0)


def meshes_kaolin2trimesh(meshes: list):
    return [trimesh.Trimesh(mesh.vertices.cpu().detach().float(),
                            mesh.faces.cpu().detach().long()) for mesh in meshes]


def meshes_trimesh2kaolin(meshes: list):
    return [TriangleMesh.from_tensors(torch.tensor(mesh.vertices).float(),
                                      torch.tensor(mesh.faces).long()) for mesh in meshes]
