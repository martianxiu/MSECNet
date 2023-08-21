import os
import sys
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
from util.MSECNet import PointcloudPatchDataset, RandomPointcloudPatchSampler, RandomPointcloudPatchSamplerDistributed 
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, Accuracy, ComputeLoss 
from util.data_util import collate_fn
from util import transform as t


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


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is None: 
        args.manual_seed = int(np.random.randint(10000, size=1))
    print(f'random seed is : {args.manual_seed}')
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False


    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_loss 
    args, best_loss = argss, 999 
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

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

    model = Model(args)
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = ComputeLoss 

    if args.optimizer == 'adamw':
        print('Use adamw\n')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        print('Use SGD\n')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr * 100, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        NotImplementedError

    if args.scheduler == 'step':
        print('Use MultiStepLR\n')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1, verbose=True)
    elif args.scheduler == 'cos':
        print('Use CosineAnealingLR\n')
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.base_lr / 100, verbose=True)
    else:
        NotImplementedError


    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.num_classes))
        logger.info(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        find_unused_parameters = True 
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=find_unused_parameters
        )

    else:
        model = torch.nn.DataParallel(model.cuda())


    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            #best_oa = 40.0
            best_oa = checkpoint['best_oa']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Augmentation
    transform_list = []
    if 'rot' in args.aug:
        # transform_list.append(t.RandomRotate(angle=[1,1,1])) # xyz rotation
        transform_list.append(t.RandomRotate(angle=[0,0,1])) # xyz rotation
    if 'jitter' in args.aug:
        transform_list.append(t.RandomJitter())
    if 'scale' in args.aug:
        transform_list.append(t.RandomScale([2./3., 3./2.], anisotropic=True))
    if 'shift' in args.aug:
        transform_list.append(t.RandomShift([0.2, 0.2, 0.2]))
    if 'pdrop' in args.aug:
        transform_list.append(t.RandomPointDrop(p=0.2, max_drop_ratio=0.8))
    print(transform_list)
    train_transform = t.Compose(transform_list)

    # dataset and loader 
    train_data = PointcloudPatchDataset(
        config=args,
        root=args.indir, # dataset folder
        shape_list_filename=args.trainset,
        patch_radius=args.patch_radius,
        points_per_patch=args.points_per_patch,
        patch_features=target_features,
        point_count_std=args.patch_point_count_std,
        seed=args.manual_seed,
        identical_epochs=args.identical_epochs,
        use_pca=args.use_pca,
        center=args.patch_center,
        point_tuple=args.point_tuple,
        cache_capacity=args.cache_capacity,
        transform=train_transform,
        sampling=args.sampling,
        split='train',
        pp_normal=args.pp_normal
    )

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = RandomPointcloudPatchSamplerDistributed(
            train_data,
            patches_per_shape=args.patches_per_shape,
            seed=args.manual_seed,
            identical_epochs=args.identical_epochs
        )
    else:
        # train_sampler = None
        train_sampler = RandomPointcloudPatchSampler(
            train_data,
            patches_per_shape=args.patches_per_shape,
            seed=args.manual_seed,
            identical_epochs=args.identical_epochs
        )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)

    val_loader = None
    if args.evaluate:
        transform_list = []
        val_transform = None
        val_data = PointcloudPatchDataset(
            config=args,
            root=args.indir, # dataset folder
            shape_list_filename=args.testset,
            patch_radius=args.patch_radius,
            points_per_patch=args.points_per_patch,
            patch_features=target_features,
            point_count_std=args.patch_point_count_std,
            seed=args.manual_seed,
            identical_epochs=args.identical_epochs,
            use_pca=args.use_pca,
            center=args.patch_center,
            point_tuple=args.point_tuple,
            cache_capacity=args.cache_capacity,
            transform=val_transform,
            sampling=args.sampling,
            split='val',
            pp_normal=args.pp_normal
        )
        if args.distributed:
            val_sampler = RandomPointcloudPatchSamplerDistributed(
                val_data,
                patches_per_shape=args.patches_per_shape,
                seed=args.manual_seed,
                identical_epochs=args.identical_epochs
            )
        else:
            val_sampler = RandomPointcloudPatchSampler(
                val_data,
                patches_per_shape=args.patches_per_shape,
                seed=args.manual_seed,
                identical_epochs=args.identical_epochs)

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        loss_train = train(train_loader, model, criterion, optimizer, epoch, output_pred_ind, output_target_ind, output_loss_weight)

        scheduler.step()

        epoch_log = epoch + 1

        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val = validate(val_loader, model, criterion, output_pred_ind, output_target_ind, output_loss_weight)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)

                is_best = loss_val< best_loss
                best_loss = min(best_loss, loss_val)
                logger.info(f'Current best loss: {best_loss:.4f}')

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_loss': best_loss, 'is_best': is_best, 'seed': args.manual_seed}, filename)
            if is_best:
                logger.info('Best validation loss updated to: {:.4f}'.format(best_loss))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest loss: %.3f' % (best_loss))


def train(train_loader, model, criterion, optimizer, epoch, output_pred_ind, output_target_ind, output_loss_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    pred_list = []
    target_list = []

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)


    for i, (coord, target, _, offset, trans) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b), (b)

        data_time.update(time.time() - end)

        coord, target, offset = coord.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        feat = target[:, 3:].contiguous() # incase there are additional features. it None, then shape is: (n, None) 
        target = target[:, :3]
        output = model(coord, feat, offset)

        loss = criterion(
            pred=output, target=target,
            outputs=args.outputs,
            output_pred_ind=output_pred_ind,
            output_target_ind=output_target_ind,
            output_loss_weight=output_loss_weight,
            patch_rot=None,
            normal_loss=args.normal_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        

        # performance metric
        pred_list.append(output)
        target_list.append(target)

        loss_meter.update(loss.item(), output.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          ))

        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)


    if main_process():
        logger.info('Train result at epoch [{}/{}]: loss {:.4f}.'.format(epoch+1, args.epochs, loss_meter.avg))
    return loss_meter.avg 

def validate(val_loader, model, criterion, output_pred_ind, output_target_ind, output_loss_weight):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    pred_list = []
    target_list = []

    model.eval()

    end = time.time()
    for i, (coord, target, _, offset, trans) in enumerate(val_loader):  # (n, 3), (n, c), (n), (b), (b)

        data_time.update(time.time() - end)

        coord, target, offset = coord.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        feat = target[:, 3:].contiguous() # incase there are additional features. it None, then shape is: (n, None) 
        target = target[:, :3]

        with torch.no_grad():
            output = model(coord, feat, offset) 

        loss = criterion(
            pred=output, target=target,
            outputs=args.outputs,
            output_pred_ind=output_pred_ind,
            output_target_ind=output_target_ind,
            output_loss_weight=output_loss_weight,
            patch_rot=None,
            normal_loss=args.normal_loss)


        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n


        pred_list.append(output)
        target_list.append(target)

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time() 



    if main_process():
        logger.info('Val result: loss {:.4f}.'.format(loss_meter.avg))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg 


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
