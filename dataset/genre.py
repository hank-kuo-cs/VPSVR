import re
import os
import torch
from glob import glob
from kaolin.rep import TriangleMesh
from PIL import Image
from torch.utils.data import Dataset
from .base import transform_image


train_classes = ['02691156', '02958343', '03001627']
test_classes = ['02828884', '03211117', '03636649', '03691459',
                '04090263', '04256520', '04379243', '04401088', '04530566']

CLASSES = {'train': train_classes, 'test': test_classes}


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
        xml_path = sample['xml_path']
        mesh_path = sample['mesh_path']

        rgb = self._load_rgb(rgb_path)
        mask = self._load_mask(mask_path)
        vertices, faces = self._load_vertices_and_faces(mesh_path)
        dist, elev, azim = self._load_view_pose(xml_path)

        return {'rgb': rgb,
                'mask': mask,
                'vertices': vertices / 1.5,  # I estimate the dist of GenRe mesh about 1.5
                'faces': faces,
                'class_id': sample['class_id'],
                'dist': dist,
                'elev': elev,
                'azim': azim}

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
                xml_paths = sorted(glob(os.path.join(dataset_path, 'genre-xml_v2', class_id, obj_id, '*.xml'))) \
                    if self.dataset_type == 'train' else [None for i in range(20)]
                mesh_paths = sorted(glob(os.path.join(dataset_path, 'objs', class_id, obj_id, '*.obj')))

                if len(rgb_paths) != 20 or len(mask_paths) != 20 or len(mesh_paths) != 20:
                    continue

                obj_count += 1

                for i in range(20):
                    self.samples.append({
                        'rgb_path': rgb_paths[i],
                        'mask_path': mask_paths[i],
                        'mesh_path': mesh_paths[i],
                        'xml_path': xml_paths[i],
                        'class_id': class_id})

                if obj_count == obj_num_each_class:
                    break

    def _load_rgb(self, rgb_path):
        is_color_jitter = True if self.dataset_type == 'train' else False
        return transform_image(Image.open(rgb_path), img_size=256,
                               color_jitter=is_color_jitter, imagenet_normalize=True)

    def _load_view_pose(self, xml_path):
        if xml_path is None:
            return 0.0, 0.0, 0.0
        xml_data = open(xml_path).read()
        view_pose = re.findall(r'lookAt origin="([0-9\.-]+), ([0-9\.-]+), ([0-9\.-]+)"', xml_data)[0]
        x, y, z = float(view_pose[0]), float(view_pose[1]), float(view_pose[2])
        dist, elev, azim = self.get_view_pose(x, y, z)
        return dist, elev, azim

    @staticmethod
    def get_view_pose(x, y, z):
        dist = torch.sqrt(torch.tensor(x ** 2 + y ** 2 + z ** 2))
        elev = 90 - torch.acos(y / dist).item() / (2 * 3.14159265359) * 360
        azim = (90 + torch.atan(torch.tensor(z / x)).item() / (2 * 3.14159265359) * 360) % 360
        return 1.5, elev, azim

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
