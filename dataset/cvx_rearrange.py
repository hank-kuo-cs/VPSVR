import os
import json
from argparse import Namespace
from glob import glob
from kaolin.rep import TriangleMesh
from PIL import Image
from torch.utils.data import Dataset
from .base import transform_image
from .genre import GenReDataset


class ConvexRearrangementDataset(Dataset):
    def __init__(self, args, dataset_type=None):
        super().__init__()

        self.args = args
        self.samples = []
        self.genre_dataset = []

        self._load_data()

    def __len__(self):
        return len(self.samples) + len(self.genre_dataset)

    def __getitem__(self, item):
        if item > len(self.samples):
            return self.genre_dataset[item - len(self.samples)]

        sample = self.samples[item]
        rgb_path, mask_path = sample['rgb_path'], sample['mask_path']
        view_path = sample['view_path']
        mesh_path = sample['mesh_path']

        rgb = self._load_rgb(rgb_path)
        mask = self._load_mask(mask_path)
        vertices, faces = self._load_vertices_and_faces(mesh_path)
        dist, elev, azim = self._load_view_pose(view_path)

        return {'rgb': rgb,
                'mask': mask,
                'vertices': vertices,
                'faces': faces,
                'dist': dist,
                'elev': elev,
                'azim': azim,
                'class_id': None}

    def _load_data(self):
        if self.args.cvx_add_genre:
            self._load_genre()

        root = self.args.root
        obj_num = self.args.size // 20

        for i in range(obj_num):
            obj_dir = os.path.join(root, '%06d' % i)

            rgb_paths = sorted(glob(os.path.join(obj_dir, '*rgb.png')))
            mask_paths = sorted(glob(os.path.join(obj_dir, '*mask.png')))
            mesh_paths = sorted(glob(os.path.join(obj_dir, '*mesh.obj')))
            view_paths = sorted(glob(os.path.join(obj_dir, 'view*.json')))

            for j in range(20):
                self.samples.append({
                    'rgb_path': rgb_paths[j],
                    'mask_path': mask_paths[j],
                    'mesh_path': mesh_paths[j],
                    'view_path': view_paths[j]
                })

    def _load_genre(self):
        genre_args = Namespace(**vars(self.args))
        genre_args.size = self.args.genre_size
        genre_args.root = self.args.genre_root
        self.genre_dataset = GenReDataset(genre_args, 'train')

    def _load_rgb(self, rgb_path):
        is_color_jitter = True
        return transform_image(Image.open(rgb_path), img_size=256,
                               color_jitter=is_color_jitter, imagenet_normalize=True)

    @staticmethod
    def _load_mask(mask_path):
        mask = transform_image(Image.open(mask_path), 256, color_jitter=False, imagenet_normalize=False)[0][None]
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        return mask

    @staticmethod
    def _load_view_pose(view_path):
        data = json.loads(open(view_path).read())
        return data['dist'], data['elev'], data['azim']


    @staticmethod
    def _load_vertices_and_faces(mesh_path):
        mesh = TriangleMesh.from_obj(mesh_path)
        return mesh.vertices, mesh.faces



