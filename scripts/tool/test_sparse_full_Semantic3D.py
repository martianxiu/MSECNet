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


def get_parser():
    parser = argparse.ArgumentParser(description='MSECNet')
    parser.add_argument('--config', type=str, default='', help='config file')
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
    # assert args.num_classes > 1
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

    # criterion = ComputeLoss 
    criterion = None 

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
        # use the seed of the best model 
        args.manual_seed = checkpoint['seed']
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    
    args.save_folder = os.path.join(args.save_folder, "Semantic3D")

    test(model,criterion, pred_dim, output_target_ind, output_pred_ind, output_loss_weight) # predict full clouds


def data_load(data_name):
    try:
        data_path = os.path.join(args.indir_Semantic3D, 'npy', data_name + '.xyz.npy')
        coord = np.load(data_path)  # N, 3 
    except:
        data_path = os.path.join(args.indir_Semantic3D, data_name + '.xyz')
        coord = np.loadtxt(data_path)
        os.makedirs(os.path.join(args.indir_Semantic3D, 'npy'), exist_ok=True)
        np.save(os.path.join(args.indir_Semantic3D, 'npy', data_name + '.xyz.npy'), coord)
    # label = np.load(label_path)  # N, 3 
    label = np.random.rand(coord.shape[0], 3) # for Semantic3D the labels (normals) are not availabel 
    feat = coord[:, 3:] # N, None; dummy data

    idx_data = []
    idx_data.append(np.arange(coord.shape[0]))
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


    check_makedirs(args.save_folder)
    
    # get all shape names in the dataset
    data_list = []
    with open(os.path.join(args.indir_Semantic3D, 'list/testset_Semantic3D_all.txt')) as f:
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
                    ) # here idx_part may contain repeated idx, which are only updated once. To avoid this,  use batch size = 1
                logger.info('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.points_per_patch, idx_part.shape[0]))


        pred = F.normalize(pred, p=2, dim=1)


        pred = pred.data.cpu().numpy() # N, 3

        # Finished a prediction for a shape. Evaluate
        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} '
                    'Name: {item} '
                    'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(idx + 1, len(data_list), label.shape[0], item=item,  batch_time=batch_time))
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


  
    
if __name__ == '__main__':
    main()
