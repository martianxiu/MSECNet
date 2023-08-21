import sys,os

import numpy as np
import torch
import torch.nn.functional as F



def ComputeLoss(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, patch_rot, normal_loss):

    loss = 0

    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            # o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            # o_target = target[output_target_ind[oi]]
            o_pred = pred
            o_target = target
            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

            if o == 'unoriented_normals':
                if 'ms_euclidean' in normal_loss:
                    loss += torch.minimum((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean() * output_loss_weight[oi]
                if 'ms_oneminuscos' in normal_loss:
                    loss += (1-torch.abs(cos_angle(o_pred, o_target))).pow(2).mean() * output_loss_weight[oi]
                if 'ms_euc_cos' in normal_loss:
                    loss += torch.minimum((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean() * 0.5 
                    loss += (1-torch.abs(cos_angle(o_pred, o_target))).pow(2).mean() * 0.5
                if 'ms_sin' in normal_loss:
                    # o_pred = F.normalize(o_pred, p=2, dim=1)
                    loss += 1.0 * torch.norm(torch.cross(o_pred, o_target, dim=-1), p=2, dim=1).mean()

            elif o == 'oriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += (o_pred-o_target).pow(2).sum(1).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1-cos_angle(o_pred, o_target)).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        else:
            raise ValueError('Unsupported output type: %s' % (o))

    return loss



def cos_angle(v1, v2):

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CrossEntropyWithSmoothing(object):
    def __init__(self, label_smoothing=0.0, ignore_index=None):
        self.eps = label_smoothing

    def __call__(self, pred, gold, weight=None):
        gold = gold.view(-1) # (batchsize, )

        if self.eps != 0.0:
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1) # (batch_size, num_class)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, weight=weight, reduction='mean')

        return loss



def Accuracy(pred_list, target_list, num_classes=15):
    pred = np.concatenate(pred_list) # b
    gt = np.concatenate(target_list) # b
    OA = np.sum(pred == gt) / (gt.shape[0])

    pred_cls_wise = [] 
    _, gt_cls_wise = np.unique(gt, return_counts=True)
    for c in range(num_classes):
        intersection = np.sum(np.logical_and(pred == c, gt == c))
        pred_cls_wise.append(intersection/gt_cls_wise[c]) 
    MA = np.mean(pred_cls_wise)
    
    return OA, MA 
    

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target



def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
