import re
import os
import torch
from glob import glob
from kaolin.rep import TriangleMesh
from PIL import Image
from torch.utils.data import Dataset
from .base import CLASSES, transform_image


class GenReDataset(Dataset):
    def __init__(self, args, dataset_type):
        super().__init__()

        self.args = args
        self.dataset_type = dataset_type
        self.samples = []

        self._load_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        rgb_path, mask_path = sample['rgb_path'], sample['mask_path']
        depth_path, mesh_path = sample['depth_path'], sample['mesh_path']

        rgb = self._load_rgb(rgb_path)
        mask = self._load_mask(mask_path)
        vertices, faces = self._load_vertices_and_faces(mesh_path)

        return {'rgb': rgb, 'mask': mask, 'vertices': vertices / 1.5, 'faces': faces}

    def _load_data(self):
        dataset_path = os.path.join(self.args.root, self.dataset_type)
        obj_num_each_class = self.args.size // len(CLASSES[self.dataset_type]) // 20

        for class_id in CLASSES[self.dataset_type]:
            img_objs_paths = sorted(glob(os.path.join(dataset_path, class_id, '*')))
            obj_ids = [re.findall(r'%s/(.+)' % os.path.join(dataset_path, class_id), img_objs_path)[0]
                       for img_objs_path in img_objs_paths]
            obj_count = 0

            for obj_id in obj_ids:
                rgb_paths = sorted(glob(os.path.join(dataset_path, class_id, obj_id, '*rgb.png')))
                mask_paths = sorted(glob(os.path.join(dataset_path, class_id, obj_id, '*silhouette.png')))
                mesh_paths = sorted(glob(os.path.join(dataset_path, 'objs', class_id, obj_id, '*.obj')))

                if len(rgb_paths) != 20 or len(mask_paths) != 20 or len(mesh_paths) != 20:
                    continue

                obj_count += 1

                for i in range(20):
                    self.samples.append({'rgb_path': rgb_paths[i],
                                         'mask_path': mask_paths[i],
                                         'mesh_path': mesh_paths[i]})

                if obj_count == obj_num_each_class:
                    break

    def _load_rgb(self, rgb_path):
        if self.dataset_type == 'train':
            return transform_image(Image.open(rgb_path), img_size=256)
        return transform_image(Image.open(rgb_path), img_size=256, color_jitter=False)

    @staticmethod
    def _load_mask(mask_path):
        mask = transform_image(Image.open(mask_path), 256, color_jitter=False, imagenet_normalize=False)[0][None]
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

        return mask

    @staticmethod
    def _load_vertices_and_faces(mesh_path):
        mesh = TriangleMesh.from_obj(mesh_path)
        return mesh.vertices, mesh.faces
