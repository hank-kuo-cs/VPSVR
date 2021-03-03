import os
import json
import torch
import random
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from kaolin.rep import TriangleMesh
from torchvision.transforms import transforms


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Original Dataset Setting
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=60000, help='0 indicates all of the dataset, '
                                                                'or it will divide equally on all classes')

    # Generate Dataset Setting
    parser.add_argument('--generate_size', type=int, default=120000, help='generate how many data')
    parser.add_argument('--generate_root', type=str, default='/eva_data/hdd1/hank/CvxRearrangement')

    return parser.parse_args()


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


args = parse_arguments()
set_seed(args.manual_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


from dataset import GenReDataset
from utils.render import PhongRenderer
from utils.transform.canonical import view2canonical, canonical2view
from utils.cvx_rearrangement.decompose import get_part_meshes
from utils.cvx_rearrangement.augment import augment_mesh
from utils.cvx_rearrangement.mesh import meshes_trimesh2kaolin, merge_parts_and_get_colors, merge_meshes


def normalize_mesh(mesh: TriangleMesh):
    mesh.vertices -= mesh.vertices.mean(0)
    mesh.vertices /= mesh.vertices.max()
    return mesh


def load_canonical_mesh(data: dict) -> trimesh.Trimesh:
    vertices, faces = data['vertices'].cuda(), data['faces'].cuda()
    dist = torch.tensor(data['dist']).view(1, 1).cuda()
    elev = torch.tensor(data['elev']).view(1, 1).cuda()
    azim = torch.tensor(data['azim']).view(1, 1).cuda()

    vertices = view2canonical(vertices[None], dist, elev, azim).view(-1, 3)

    return trimesh.Trimesh(vertices.cpu().detach().float(), faces.cpu().detach().long())


def transform2view_center(points: torch.Tensor, dist: float, elev: float, azim: float):
    points = points[None].cuda()
    dist = torch.tensor(dist).view(1, 1).cuda().float()
    elev = torch.tensor(elev).view(1, 1).cuda().float()
    azim = torch.tensor(azim).view(1, 1).cuda().float()
    points = canonical2view(points, dist, elev, azim)
    return points[0].cpu().detach()


def get_random_view_pose():
    dist = random.uniform(4, 6)
    elev = random.randint(15, 60)
    azim = random.randint(0, 360)

    return dist, elev, azim


def get_random_light_x():
    return random.uniform(-10, 10)


def rearrange_two_data(data1, data2, save_dir_path: str):
    to_pil = transforms.ToPILImage()

    mesh1 = normalize_mesh(load_canonical_mesh(data1))  # TriMesh
    mesh2 = normalize_mesh(load_canonical_mesh(data2))

    part_meshes1, symmetry_indices1 = get_part_meshes(mesh1, data1['class_id'])  # list(TriMesh, ...), list(int, ...)
    part_meshes2, symmetry_indices2 = get_part_meshes(mesh2, data2['class_id'])

    aug_mesh1 = augment_mesh(part_meshes1, symmetry_indices1, data1['class_id'])  # list(TriangleMesh, ...)
    aug_mesh2 = augment_mesh(part_meshes2, symmetry_indices2, data2['class_id'])

    merged_mesh = merge_meshes([aug_mesh1, aug_mesh2])  # TriMesh
    convex_hulls = merged_mesh.convex_decomposition(24)
    if not isinstance(convex_hulls, list):
        convex_hulls = [convex_hulls]

    final_part_meshes = meshes_trimesh2kaolin(convex_hulls)
    final_mesh, uv, texture = merge_parts_and_get_colors(final_part_meshes)

    for j in range(20):
        dist, elev, azim = get_random_view_pose()
        light_direction = torch.tensor([[get_random_light_x(), 5, 10]], dtype=torch.float).cuda()

        rgb, mask, _ = PhongRenderer.render_single_image_with_single_mesh(
            final_mesh, dist, elev, azim, uv, texture, light_direction=light_direction)

        rgb = rgb[0].cpu().permute(2, 0, 1)
        mask = mask[0].cpu().permute(2, 0, 1)

        view_center_vertices = transform2view_center(final_mesh.vertices, dist, elev, azim)
        mesh = TriangleMesh.from_tensors(view_center_vertices, final_mesh.faces.detach().cpu())

        rgb = to_pil(rgb)
        mask = to_pil(mask)
        json_data = json.dumps({'dist': dist, 'elev': elev, 'azim': azim})

        rgb.save(os.path.join(save_dir_path, '%02d_rgb.png' % j))
        mask.save(os.path.join(save_dir_path, '%02d_mask.png' % j))
        mesh.save_mesh(os.path.join(save_dir_path, '%02d_mesh.obj' % j))

        with open(os.path.join(save_dir_path, 'view_pose_%02d.json' % j), 'w') as f:
            f.write(json_data)
            f.close()


def generate():
    dataset = GenReDataset(args, 'train')

    for i in tqdm(range(args.generate_size // 20)):
        save_dir_path = os.path.join(args.generate_root, '%06d' % i)
        os.makedirs(save_dir_path, exist_ok=True)

        while True:
            try:
                indices = random.sample([i for i in range(len(dataset))], 2)

                data1 = dataset[indices[0]]
                data2 = dataset[indices[1]]

                rearrange_two_data(data1, data2, save_dir_path)
                break
            except Exception as e:
                print(e)


if __name__ == '__main__':
    generate()
