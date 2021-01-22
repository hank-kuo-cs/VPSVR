import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import MSELoss
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Testing Setting
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='use which model to evaluate')

    # Dataset Setting
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=0, help='0 indicates all of the dataset, '
                                                            'or it will divide equally on all classes')

    # Record Setting
    parser.add_argument('--record_batch_interval', type=int, default=20, help='record prediction result every N batch')

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


from dataset import GenReDataset, R2N2Dataset, collate_func, CLASS_DICT
from model import DepthEstimationNet
from utils.render import DepthRenderer
from utils.visualize import save_depth_result


def set_path(args):
    checkpoint_path = './checkpoint/den/den_epoch%03d.pth' % args.epoch

    record_paths = {'loss': './output/loss/eval/epoch%.3d' % args.epoch,
                    'depth': './output/depth/eval/epoch%.3d' % args.epoch}
    for record_path in list(record_paths.values()):
        os.makedirs(record_path, exist_ok=True)

    return checkpoint_path, record_paths


@torch.no_grad()
def eval(args):
    dataset = GenReDataset(args, 'test') if args.dataset == 'genre' else R2N2Dataset(args, 'test')
    print('Load %s training dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False, collate_fn=collate_func)
    checkpoint_path, record_paths = set_path(args)

    den = DepthEstimationNet().cuda()
    den.load_state_dict(torch.load(checkpoint_path))
    den.eval()

    mse_loss_func = MSELoss()
    class_losses = {}
    class_n = {}

    progress_bar = tqdm(enumerate(dataloader))

    for idx, data in progress_bar:
        # Load data
        rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
        class_ids = data['class_id']
        vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
        faces = [one_faces.cuda() for one_faces in data['faces']]  # list((F1, 2), ..., (FB, 2))

        # Predict Depth
        gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces)
        predict_depths = den(rgbs)

        batch_size = rgbs.size(0)
        for b in range(batch_size):
            depth_loss = mse_loss_func(predict_depths[b][None], gt_depths[b][None])
            class_id = class_ids[b]

            if class_id in class_losses:
                class_losses[class_id] += depth_loss
                class_n[class_id] += 1
            else:
                class_losses[class_id] = depth_loss
                class_n[class_id] = 1

        if (idx + 1) % args.record_batch_interval == 0:
            depth_save_path = os.path.join(record_paths['depth'], 'batch%d.png' % (idx + 1))
            save_depth_result(rgbs[0], predict_depths[0], gt_depths[0], depth_save_path)

    avg_loss = 0.0
    print('Epoch %d' % args.epoch)
    print('id\t\tloss\t\tname')

    for k in list(class_losses.keys()):
        class_losses[k] = (class_losses[k] / class_n[k]).item()
        print('%s\t%.6f\t%s' % (k, class_losses[k], CLASS_DICT[k]))
        avg_loss += class_losses[k]

    avg_loss /= len(list(class_losses.keys()))
    print('total mean mse loss = %.6f' % avg_loss)
    class_losses['total'] = avg_loss

    np.savez(os.path.join(record_paths['loss'], 'mse.npz'), **class_losses)


if __name__ == '__main__':
    eval(args)
