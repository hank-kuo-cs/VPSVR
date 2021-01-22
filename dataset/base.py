import torch
from PIL import Image
from torchvision.transforms import transforms


CLASS_DICT = {'02691156': 'airplane', '04090263': 'rifle', '03211117': 'display', '04379243': 'table',
              '04401088': 'telephone', '02958343': 'car', '03001627': 'chair', '02828884': 'bench', '03636649': 'lamp',
              '02933112': 'cabinet', '03691459': 'loudspeaker', '04256520': 'sofa', '04530566': 'watercraft'}


def collate_func(batch_data):
    check_data_keys(batch_data[0])
    collate_data = {'rgb': [], 'mask': [],
                    'vertices': [], 'faces': [],
                    'class_id': [],
                    'dist': [], 'elev': [], 'azim': []}

    for i in range(len(batch_data)):
        for key in list(collate_data.keys()):
            if key in ['rgb', 'mask']:
                collate_data[key].append(batch_data[i][key][None])
            elif key in ['dist', 'elev', 'azim']:
                collate_data[key].append(torch.tensor(batch_data[i][key], dtype=torch.float)[None])
            else:
                collate_data[key].append(batch_data[i][key])

    for key in ['rgb', 'mask', 'dist', 'elev', 'azim']:
        collate_data[key] = torch.cat(collate_data[key])

    return collate_data


def transform_image(image: Image, img_size: int, color_jitter: bool = True, imagenet_normalize: bool = True):
    transform_list = [transforms.Resize(img_size)]

    if color_jitter:
        transform_list.append(transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4))

    transform_list.append(transforms.ToTensor())

    if imagenet_normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)(image)


def check_data_keys(data):
    for key in ['rgb', 'mask', 'vertices', 'faces', 'class_id', 'dist', 'elev', 'azim']:
        if key not in data:
            raise AssertionError('The key "%s" should be in sample data' % key)
