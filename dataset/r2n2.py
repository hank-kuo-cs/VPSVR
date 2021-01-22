import re
import os
import torch
from glob import glob
from tqdm import tqdm
from kaolin.rep import TriangleMesh
from PIL import Image
from torch.utils.data import Dataset
from .base import transform_image


train_classes = ['02691156', '02958343', '03001627']
test_classes = ['02828884', '03211117', '03636649', '03691459', '02933112'
                '04090263', '04256520', '04379243', '04401088', '04530566']

CLASSES = {'train': train_classes, 'test': test_classes}


class R2N2Dataset(Dataset):
    def __init__(self, args, dataset_type):
        super().__init__()

        self.args = args
        self.dataset_type = dataset_type
        self.split_data = {}
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

        dataset_indices = self.split_data[self.dataset_type]
        obj_num_each_class = self.args.size // len(CLASSES[self.dataset_type]) // 24
        class_obj_num = {}
        for class_id in CLASSES[self.dataset_type]:
            class_obj_num[class_id] = 0

        for i in tqdm(range(len(dataset_indices))):
            class_id, obj_id = dataset_indices[i][0], dataset_indices[i][1]
            if class_id not in class_obj_num or class_obj_num[class_id] == obj_num_each_class:
                continue

            imgs_dir_path = os.path.join(self.args.root, class_id, obj_id, 'rendering')
            imgs_path, azims, elevs, dists = self._load_imgs_in_one_dir(imgs_dir_path)

            meshs_path = sorted(glob(os.path.join(self.args.root, class_id, obj_id, 'objs') + '/*.obj'))

            for j in range(len(imgs_path)):
                self.samples.append({'img_path': imgs_path[j], 'mesh_path': meshs_path[j], 'class_id': class_id,
                                     'dist': dists[j], 'elev': elevs[j], 'azim': azims[j]})

            class_obj_num[class_id] += 1

    def _load_split_data(self):
        split_data = {}
        str_data = open(self.args.root + '/split.csv', 'r').read()

        split_data['train'] = re.findall(r'.+,(.+),.+,(.+),train', str_data)
        split_data['test'] = re.findall(r'.+,(.+),.+,(.+),test', str_data)
        split_data['train'].extend(re.findall(r'.+,(.+),.+,(.+),val', str_data))

        self.split_data = split_data

    def _load_imgs_in_one_dir(self, dir_path):
        meta_path = dir_path + '/rendering_metadata.txt'
        imgs_path = sorted(glob(dir_path + '/*.png'))
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
        meta_datas = re.findall(r'([\.0-9]+?) ([\.0-9]+?) 0 ([\.0-9]+?) 25', meta_str)
        for meta_data in meta_datas:
            cameras = list(map(float, meta_data))
            cameras[2] *= 1.754

            azims.append(cameras[0])
            elevs.append(cameras[1])
            dists.append(cameras[2])

        return azims, elevs, dists

    @staticmethod
    def _load_vertices_and_faces(mesh_path):
        mesh = TriangleMesh.from_obj(mesh_path)
        return mesh.vertices, mesh.faces


