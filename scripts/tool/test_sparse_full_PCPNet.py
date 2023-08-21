import os
from os.path import join, exists
import sys
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs, Accuracy, ComputeLoss 
from util.MSECNet import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialPointcloudPatchSampler
from util import transform as t
from util.data_util import collate_fn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_parser():
    parser = argparse.ArgumentParser(description='MSECNet')
    parser.add_argument('--config', type=str, default='config/pcpnet/config.yaml', help='config file')
    parser.add_argument('opts', help='', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))

    # model 
    if args.arch == 'MSECNet':
        from architectures import MSECNet as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    
    # update the output dim based on the #local properties
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in args.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        else:
            raise ValueError('Unknown output: %s' % (o))
    args.num_classes = pred_dim


    model = Model(args).cuda()
    logger.info(model)
    logger.info(f'Number of parameters (M): {count_parameters(model)/1000/1000}')

    criterion = ComputeLoss 

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
        args.manual_seed = int(checkpoint['seed'])
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    

    test(model,criterion, pred_dim, output_target_ind, output_pred_ind, output_loss_weight) # predict full clouds
    evaluate(args.indir, args.save_folder, args.save_folder) # evaluate sparse patch performance


def data_load(data_name):
    data_path = os.path.join(args.indir, data_name + '.xyz.npy')
    label_path = os.path.join(args.indir, data_name + '.normals.npy')
    try:
        coord = np.load(data_path)  # N, 3 
    except:
        coord = np.loadtxt(data_path[:-4])
        np.save(data_path, coord)
    try:
        label = np.load(label_path)  # N, 3 
    except:
        label = np.loadtxt(label_path[:-4])
        np.save(label_path, label)

    feat = coord[:, 3:] # N, None; dummy data

    idx_data = []
    idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def prepare_input(xyz, feat):
    # the first point is the center due to kNN 
    xyz = xyz - xyz[0] # centralize
    max_dist = np.max(np.sum(xyz ** 2, axis=1)**0.5, axis=0)
    xyz = xyz / max_dist 
    return xyz, feat

def test(model, criterion, pred_dim, output_target_ind, output_pred_ind, output_loss_weight):
    logger.info('>>>>>>>>>>>>>>>> Start Full Shape Prediction >>>>>>>>>>>>>>>>')
    model.eval()
    batch_time = AverageMeter()

    args.save_folder = os.path.join(args.save_folder, args.data_name)

    check_makedirs(args.save_folder)
    
    # get all shape names in the dataset
    data_list = []
    with open(os.path.join(args.indir, 'testset_all.txt')) as f:
        data_list = f.readlines()
    data_list= [x.strip() for x in data_list]
    data_list= list(filter(None, data_list))

    pred_save, label_save = [], []

    total_time = 0 # model inference time for all shapes
    # per-shape prediction 
    for idx, item in enumerate(data_list):
        shape_time = 0 # model inference time for a shape
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, '{}_normal_est.npy'.format(item))
        label_save_path = os.path.join(args.save_folder, '{}_normal_gt.npy'.format(item))
        if os.path.isfile(pred_save_path):
            logger.info('{}/{}: {}, loaded normal.'.format(idx + 1, len(data_list), item))
            pred = torch.FloatTensor(np.load(pred_save_path, allow_pickle=True)).cuda(non_blocking=True)
            label = np.load(label_save_path, allow_pickle=True)
        else:
            coord, feat, label, idx_data = data_load(item) # load a shape with full points 
            pred = torch.zeros(label.shape).cuda() # container 
            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list, trans_list  = [], [], [], [], []
            if args.use_pca:
                pca = t.PCARotate(return_trans=True)
            for i in range(idx_size):
                logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
                idx_part = idx_data[i]
                coord_part, feat_part = coord[idx_part], feat[idx_part]
                coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0 # containers for potential based predictions 
                # generating batches
                while idx_uni.size != idx_part.shape[0]: # looks like a soft farthest cropping. Must cover all index (points). 
                    init_idx = np.argmin(coord_p) # random sampling. just choose the smallest index  
                    dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1) # squared dist of all points to selected point 
                    idx_crop = np.argsort(dist)[:args.points_per_patch] # take kNN using the query point as a center. 
                    coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                    coord_sub, feat_sub = prepare_input(coord_sub, feat_sub)
                    if args.use_pca:
                        coord_sub, _, _, trans_sub = pca(coord_sub, np.random.rand(coord_sub.shape[0], 3), None) 
                    else:
                        trans_sub = np.eye(3,3).astype(np.float32)
                    idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size), trans_list.append(trans_sub)
                    idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))

                    # potential update
                    idx_update = idx_crop[:int(args.points_per_patch * 0.7)] # do not increment points that are too far from the center. 
                    # idx_update = idx_crop
                    dist = dist[idx_update]  
                    delta = np.square(1 - dist / np.max(dist)) # farther poitns have smaller delta    
                    coord_p[idx_update] += delta # increment the random index. so that next index will be quite far from this one. 
                

            args.batch_size_test = 128 
            smoothing = 0.9
            batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                idx_part, coord_part, feat_part, offset_part, trans_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i], trans_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                trans_part = torch.FloatTensor(np.stack(trans_part, 0)).cuda(non_blocking=True) # b, 3, 3
                start_time = time.time()
                with torch.no_grad():
                    pred_part = model(coord_part, feat_part, offset_part)  # (bn, 3)
                end_time = time.time()
                elapsed_time = end_time - start_time
                shape_time += elapsed_time
                total_time += elapsed_time

                if args.use_pca:
                    pred_part = pred_part.view(-1, args.points_per_patch, 3) # b, n, 3
                    pred_part[:, :] = torch.bmm(pred_part, trans_part.transpose(2, 1)) # b, n, 3
                    pred_part = pred_part.view(-1, 3) # bn, 3
                torch.cuda.empty_cache()

                pred_part = pred_part.view(-1, args.points_per_patch, 3)
                idx_part = idx_part.reshape(-1, args.points_per_patch)
                for idx_update, pred_update  in zip(idx_part, pred_part):
                    pred[idx_update, :] = smoothing * pred[idx_update, :] + (1-smoothing)* torch.where(
                        (pred[idx_update, :] - pred_update).pow(2).sum(1, keepdim=True) - (pred[idx_update, :] + pred_update).pow(2).sum(1, keepdim=True) > 0,
                        - pred_update,
                        pred_update
                    ) 
                logger.info('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.points_per_patch, idx_part.shape[0]))
        pred = F.normalize(pred, p=2, dim=1)

        loss = criterion(
            pred=pred, target=torch.FloatTensor(label).cuda(non_blocking=True),
            outputs=args.outputs,
            output_pred_ind=output_pred_ind,
            output_target_ind=output_target_ind,
            output_loss_weight=output_loss_weight,
            patch_rot=None,
            normal_loss=args.normal_loss)

        pred = pred.data.cpu().numpy() # N, 3

        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} '
                    'Name: {item} '
                    'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss:.3f}  '.format(idx + 1, len(data_list), label.shape[0], item=item,  batch_time=batch_time, loss=loss))
        logger.info(f'Shape {item} takes {shape_time:.3f} sec for inference. Batchsize is {args.batch_size_test}. Number of input point is {args.points_per_patch}')
        # save  
        pred_save.append(pred)
        label_save.append(label)
        np.save(pred_save_path, pred)
        np.save(label_save_path, label)

    logger.info(f'Takes {total_time:.3f} sec for inference ({total_time/108:.3f} sec per shape.)')

    # save all prediction
    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "normals.pickle"), 'wb') as handle:
        pickle.dump({'normals': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def normal_RMSE(normal_gts, normal_preds, eval_file='log.txt'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()

    rms   = []
    rms_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5  = []
    pgp_alpha = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented rms
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        ### Error metric
        rms.append(np.sqrt(np.mean(np.square(ang))))
        ### Portion of good points
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))

        pgp_alpha.append(pgp_alpha_shape)

        # Oriented rms
        rms_o.append(np.sqrt(np.mean(np.square(np.rad2deg(np.arccos(nn))))))


    avg_rms   = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    log_file.close()

    return avg_rms

def evaluate(normal_gt_path, normal_pred_path, output_dir):
    logger.info('>>>>>>>>>>>>>>>> Start Sparse Patch Evaluation >>>>>>>>>>>>>>>>')
    eval_summary_dir = os.path.join(args.save_folder, 'summary')
    os.makedirs(eval_summary_dir, exist_ok=True)
    args.eval_list = ['testset_no_noise.txt', 
                      'testset_low_noise.txt', 
                      'testset_med_noise.txt', 
                      'testset_high_noise.txt',
                      'testset_vardensity_striped.txt', 
                      'testset_vardensity_gradient.txt'] 

    all_avg_rms = []
    for cur_list in args.eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, cur_list)
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all shapes
        normal_gts = []
        normal_preds = []
        for shape in shape_names:
            print(shape)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal_est.npy'))                  # (n, 3)
            normal_gt = np.load(os.path.join(normal_gt_path, shape + '.normals.npy'))                  # (n, 3)
            points_idx = np.load(os.path.join(normal_gt_path, shape + '.pidx.npy'))      # (n,)
            normal_gt = normal_gt[points_idx, :] # extract sparse patche 
            normal_pred = normal_pred[points_idx, :]

            normal_gts.append(normal_gt)
            normal_preds.append(normal_pred)

        ### compute RMSE per-list
        avg_rms = normal_RMSE(normal_gts=normal_gts,
                            normal_preds=normal_preds,
                            eval_file=os.path.join(eval_summary_dir, cur_list[:-4] + '_evaluation_results.txt'))
        all_avg_rms.append(avg_rms)
        print('RMSE: %f' % avg_rms)

    s = '\n {} \n All RMS not oriented (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_rms), np.mean(all_avg_rms))
    print(s)

    
if __name__ == '__main__':
    main()
