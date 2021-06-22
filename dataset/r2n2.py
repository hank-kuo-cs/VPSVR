import re
import os
import torch
from glob import glob
from tqdm import tqdm
from kaolin.rep import TriangleMesh
from PIL import Image
from torch.utils.data import Dataset
from .base import transform_image


class R2N2Dataset(Dataset):
    def __init__(self, args, dataset_type):
        super().__init__()

        self.args = args
        self.dataset_type = dataset_type
        self.p2m_data = []
        self.samples = []

        self._load_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        img_path, mesh_path = sample['img_path'], sample['mesh_path']
        rgb, mask = self._load_rgb_and_mask(img_path)
        vertices, faces = self._load_vertices_and_faces(mesh_path)

        return {'rgb': rgb,
                'mask': mask,
                'vertices': vertices,
                'faces': faces,
                'class_id': sample['class_id'],
                'dist': sample['dist'],
                'elev': sample['elev'],
                'azim': sample['azim']}

    def _load_data(self):
        self._load_split_data()

        obj_num_each_class = self.args.size // 13 // 5
        class_obj_nums = {}

        for i in tqdm(range(len(self.p2m_data))):
            class_id, obj_id = self.p2m_data[i][0], self.p2m_data[i][1]
            if class_id not in class_obj_nums:
                class_obj_nums[class_id] = 0
            if class_obj_nums[class_id] == obj_num_each_class and obj_num_each_class > 0:
                continue

            imgs_dir_path = os.path.join(self.args.root, class_id, obj_id, 'rendering')
            imgs_path = sorted(glob(imgs_dir_path + '/*.png'))[:5]
            # imgs_path, _, _, _ = self._load_imgs_in_one_dir(imgs_dir_path)

            meshes_path = sorted(glob(os.path.join(self.args.root, class_id, obj_id, 'objs') + '/*.obj'))[:5]

            if not len(meshes_path):
                continue

            for j in range(len(imgs_path)):
                self.samples.append({'img_path': imgs_path[j], 'mesh_path': meshes_path[j], 'class_id': class_id,
                                     'dist': 1.0, 'elev': 0.0, 'azim': 0.0})

            class_obj_nums[class_id] += 1

    def _load_split_data(self):
        str_data = open(os.path.join(self.args.root, 'p2m_%s.txt' % self.dataset_type), 'r').read()
        self.p2m_data = re.findall(r'ShapeNetP2M/(.+)/(.+)/rendering/00.dat', str_data)

    def _load_imgs_in_one_dir(self, dir_path):
        meta_path = dir_path + '/rendering_metadata.txt'
        imgs_path = sorted(glob(dir_path + '/*.png'))[:5]
        azims, elevs, dists = self._load_meta(meta_path)

        return imgs_path, azims, elevs, dists

    def _load_rgb_and_mask(self, img_path: str) -> (torch.Tensor, torch.Tensor):
        img = Image.open(img_path)
        rgb, mask = img.convert('RGB'), img.split()[3]

        rgb = transform_image(rgb, 256, color_jitter=(self.dataset_type == 'train'))
        mask = transform_image(mask, 256, color_jitter=False, imagenet_normalize=False)

        return rgb, mask

    @staticmethod
    def _load_meta(meta_path: str) -> (list, list, list):
        azims, elevs, dists = [], [], []
        meta_str = open(meta_path, 'r').read()
        meta_datas = re.findall(r'([\.0-9]+?) ([\.0-9]+?) 0 ([\.0-9]+?) 25', meta_str)[:5]
        for meta_data in meta_datas:
            cameras = list(map(float, meta_data))
            cameras[2] *= 1.754

            azims.append(cameras[0])
            elevs.append(cameras[1])
            # dists.append(cameras[2])
            dists.append(1.0)

        return azims, elevs, dists

    @staticmethod
    def _load_vertices_and_faces(mesh_path):
        mesh = TriangleMesh.from_obj(mesh_path)
        return mesh.vertices, mesh.faces
