import numpy as np
import random
import torch



def collate_fn(batch):
    coord, feat, label, trans = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.LongTensor(label), torch.IntTensor(offset), torch.stack(trans, 0)

