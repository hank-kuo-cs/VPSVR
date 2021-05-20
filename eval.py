import os
import torch
import random
import argparse
import numpy as np
from kaolin.rep import TriangleMesh
from tqdm import tqdm
from torch.nn import MSELoss
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Evaluation Setting
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='use which epoch to test,'
                                                              'it will be ignore if below model paths given')
    parser.add_argument('--depth_ae_path', type=str, default='')
    parser.add_argument('--depth_en_path', type=str, default='')
    parser.add_argument('--translate_de_path', type=str, default='')
    parser.add_argument('--volume_rotate_de_path', type=str, default='')
    parser.add_argument('--deform_gcn_path', type=str, default='')
    parser.add_argument('--use_gt_depth', action='store_true')

    # Dataset Setting
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=0, help='0 indicates all of the dataset, '
                                                            'or it will divide equally on all classes')

    # Volumetric Primitive
    parser.add_argument('--sphere_num', type=int, default=16, help='number of spheres')
    parser.add_argument('--cuboid_num', type=int, default=0, help='number of cuboids')
    parser.add_argument('--vertex_num', type=int, default=128, help='number of vertices of each primitive')

    # Record Setting
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--record_batch_interval', type=int, default=20, help='record prediction result every N batch')
    parser.add_argument('--save_mesh', action='store_true', help='save the all mesh result')

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
from model import DepthEstimationUNet
from model.two_step import DepthEncoder, TranslateDecoder, VolumeRotateDecoder, DeformGCN
from utils.sampling import Sampling
from utils.meshing import Meshing
from utils.loss import ChamferDistanceLoss
from utils.render import DepthRenderer
from utils.perceptual import get_local_features
from utils.visualize import save_depth_result, save_vp_result


def load_model(args):
    vp_num = args.cuboid_num + args.sphere_num
    depth_ae_path = 'checkpoint/depth_ae/depth_ae_epoch%03d.pth' % args.epoch \
        if not args.depth_ae_path else args.depth_ae_path
    depth_ae = DepthEstimationUNet().cuda()
    depth_ae.load_state_dict(torch.load(depth_ae_path))

    depth_en_path = 'checkpoint/depth_en/depth_en_epoch%03d.pth' % args.epoch \
        if not args.depth_en_path else args.depth_en_path
    depth_en = DepthEncoder().cuda()
    depth_en.load_state_dict(torch.load(depth_en_path))

    translate_de_path = 'checkpoint/translate_de/translate_de_epoch%03d.pth' % args.epoch \
        if not args.translate_de_path else args.translate_de_path
    translate_de = TranslateDecoder(vp_num=vp_num).cuda()
    translate_de.load_state_dict(torch.load(translate_de_path))

    volume_rotate_de_path = 'checkpoint/volume_rotate_de/volume_rotate_de_epoch%03d.pth' % args.epoch \
        if not args.volume_rotate_de_path else args.volume_rotate_de_path
    volume_rotate_de = VolumeRotateDecoder().cuda()
    volume_rotate_de.load_state_dict(torch.load(volume_rotate_de_path))

    deform_gcn_path = 'checkpoint/deform_gcn/deform_gcn_epoch%03d.pth' % args.epoch \
        if not args.deform_gcn_path else args.deform_gcn_path
    deform_gcn = DeformGCN(v_num=args.vertex_num * vp_num).cuda()
    deform_gcn.load_state_dict(torch.load(deform_gcn_path))

    return depth_ae, depth_en, translate_de, volume_rotate_de, deform_gcn


def set_path(args):
    output_path = './output/eval/epoch%03d' % args.epoch if not args.output_path else args.output_path
    if args.use_gt_depth:
        output_path += '-oracle'
    record_paths = {'loss': os.path.join(output_path, 'loss'),
                    'depth': os.path.join(output_path, 'depth'),
                    'vp': os.path.join(output_path, 'vp'),
                    'mesh': os.path.join(output_path, 'mesh')}
    for record_path in list(record_paths.values()):
        os.makedirs(record_path, exist_ok=True)

    return record_paths


@torch.no_grad()
def eval(args):
    dataset = GenReDataset(args, 'test') if args.dataset == 'genre' else R2N2Dataset(args, 'test')
    print('Load %s testing dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False, collate_fn=collate_func)

    vp_num = args.cuboid_num + args.sphere_num
    record_paths = set_path(args)
    depth_ae, depth_en, translate_de, volume_rotate_de, deform_gcn = load_model(args)

    depth_ae.eval()
    depth_en.eval()
    translate_de.eval()
    volume_rotate_de.eval()
    deform_gcn.eval()

    mse_loss_func, cd_loss_func = MSELoss(), ChamferDistanceLoss()

    class_losses = {'depth': {}, 'vp_cd': {}, 'mesh_cd': {}}
    class_n = {}

    progress_bar = tqdm(enumerate(dataloader))

    for idx, data in progress_bar:
        rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
        class_ids = data['class_id']
        vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
        faces = [one_faces.cuda() for one_faces in data['faces']]

        gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces, normalize=True)
        gt_meshes = Meshing.meshing_vertices_faces(vertices, faces)
        gt_points = Sampling.sample_mesh_points(gt_meshes, sample_num=1024)

        rgbs *= masks
        pred_depths = depth_ae(rgbs)
        pred_depths *= masks

        input_depths = gt_depths if args.use_gt_depth else pred_depths

        global_features, perceptual_feature_list = depth_en(input_depths)

        translates = translate_de(global_features)
        vp_center_points = torch.cat([t[:, None, :] for t in translates], 1)  # (B, K, 3)

        volumes, rotates = [], []
        vp_features = get_local_features(vp_center_points, input_depths, perceptual_feature_list)  # (B, K, F)

        for i in range(vp_num):
            one_vp_feature = vp_features[:, i, :]  # (B, F)
            volume, rotate = volume_rotate_de(one_vp_feature)

            volumes.append(volume)
            rotates.append(rotate)

        pred_meshes = Meshing.vp_meshing(volumes, rotates, translates,
                                         cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)
        vp_meshes = [TriangleMesh.from_tensors(m.vertices.clone(), m.faces.clone()) for m in pred_meshes]
        pred_coarse_points = Sampling.sample_mesh_points(vp_meshes, sample_num=1024)

        vp_cd_loss = cd_loss_func(pred_coarse_points, gt_points, each_batch=True)[0]

        deformation = deform_gcn(pred_meshes, input_depths, perceptual_feature_list, global_features)

        for b in range(len(pred_meshes)):
            pred_meshes[b].vertices += deformation[b]

        pred_fine_points = Sampling.sample_mesh_points(pred_meshes, sample_num=1024)

        mesh_cd_loss = cd_loss_func(pred_fine_points, gt_points, each_batch=True)[0]

        batch_size = rgbs.size(0)
        for b in range(batch_size):
            depth_loss = mse_loss_func(pred_depths[b][None], gt_depths[b][None])
            class_id = class_ids[b]

            if class_id in class_losses['depth']:
                class_losses['depth'][class_id] += depth_loss
                class_losses['vp_cd'][class_id] += vp_cd_loss[b]
                class_losses['mesh_cd'][class_id] += mesh_cd_loss[b]
                class_n[class_id] += 1
            else:
                class_losses['depth'][class_id] = depth_loss
                class_losses['vp_cd'][class_id] = vp_cd_loss[b]
                class_losses['mesh_cd'][class_id] = mesh_cd_loss[b]
                class_n[class_id] = 1

        if (idx + 1) % args.record_batch_interval == 0:
            depth_save_path = os.path.join(record_paths['depth'], 'batch%d.png' % (idx + 1))
            vp_save_path = os.path.join(record_paths['vp'], 'batch%d.png' % (idx + 1))
            mesh_save_path = os.path.join(record_paths['mesh'], 'batch%d.png' % (idx + 1))

            save_depth_result(rgb=rgbs[0], mask=masks[0],
                              predict_depth=pred_depths[0], gt_depth=gt_depths[0], save_path=depth_save_path)
            save_vp_result(rgb=rgbs[0], mask=masks[0], input_depth=input_depths[0],
                           predict_mesh=vp_meshes[0], gt_mesh=gt_meshes[0],
                           vp_num=vp_num, vertex_num=args.vertex_num, save_path=vp_save_path)
            save_vp_result(rgb=rgbs[0], mask=masks[0], input_depth=input_depths[0],
                           predict_mesh=pred_meshes[0], gt_mesh=gt_meshes[0],
                           vp_num=vp_num, vertex_num=args.vertex_num, save_path=mesh_save_path)

    avg_depth_loss, avg_vp_cd_loss, avg_mesh_cd_loss = 0.0, 0.0, 0.0

    print('Depth MSE Loss')
    print('id\t\tloss\t\tname')

    for k in list(class_losses['depth'].keys()):
        class_losses['depth'][k] = (class_losses['depth'][k] / class_n[k]).item()
        print('%s\t%.6f\t%s' % (k, class_losses['depth'][k], CLASS_DICT[k]))
        avg_depth_loss += class_losses['depth'][k]

    avg_depth_loss /= len(list(class_losses['depth'].keys()))
    print('total mean depth mse loss = %.6f' % avg_depth_loss)
    class_losses['depth']['total'] = avg_depth_loss

    np.savez(os.path.join(record_paths['loss'], 'depth.npz'), **class_losses['depth'])

    print('VP CD Loss')
    print('id\t\tloss\t\tname')

    for k in list(class_losses['vp_cd'].keys()):
        class_losses['vp_cd'][k] = (class_losses['vp_cd'][k] / class_n[k]).item()
        print('%s\t%.6f\t%s' % (k, class_losses['vp_cd'][k], CLASS_DICT[k]))
        avg_vp_cd_loss += class_losses['vp_cd'][k]

    avg_vp_cd_loss /= len(list(class_losses['vp_cd'].keys()))
    print('total mean vp cd loss = %.6f' % avg_vp_cd_loss)
    class_losses['vp_cd']['total'] = avg_vp_cd_loss

    np.savez(os.path.join(record_paths['loss'], 'vp_cd.npz'), **class_losses['vp_cd'])

    print('Mesh CD Loss')
    print('id\t\tloss\t\tname')

    for k in list(class_losses['mesh_cd'].keys()):
        class_losses['mesh_cd'][k] = (class_losses['mesh_cd'][k] / class_n[k]).item()
        print('%s\t%.6f\t%s' % (k, class_losses['mesh_cd'][k], CLASS_DICT[k]))
        avg_mesh_cd_loss += class_losses['mesh_cd'][k]

    avg_mesh_cd_loss /= len(list(class_losses['mesh_cd'].keys()))
    print('total mean mesh cd loss = %.6f' % avg_mesh_cd_loss)
    class_losses['mesh_cd']['total'] = avg_mesh_cd_loss

    np.savez(os.path.join(record_paths['loss'], 'mesh_cd.npz'), **class_losses['mesh_cd'])


if __name__ == '__main__':
    eval(args)
