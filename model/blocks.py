import numpy as np
from lib.pointops.functions import pointops
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

def block_decider(name):
    if name == 'pointwise_mlp':
        return PointWiseMLP 
    if name == 'pointnet2':
        return PointNet2

    if name == 'simple':
        return SimpleBlock 
    if name == 'residual':
        return ResidualBlock
    if name == 'residual_fusion':
        return ResidualFusionBlock 
    
    if name == 'upsample':
        return Upsampling 
    if name == 'downsample':
        return Downsampling 

    if name == 'adaptive_laplacian':
        return AdaptiveLaplacian 
    if name == 'edge_conditioning':
        return EdgeConditioning 

class PointWiseMLP(nn.Module):
    def __init__(self, d_in, d_out, config):
        super().__init__()
        d_mid = d_in = d_out
        self.mlp = nn.Sequential(
            nn.Linear(d_mid, d_mid),
            nn.BatchNorm1d(d_mid),
            nn.ReLU(inplace=True),
            nn.Linear(d_mid, d_mid),
            nn.BatchNorm1d(d_mid),
            nn.ReLU(inplace=True),
            nn.Linear(d_mid, d_mid),
            nn.BatchNorm1d(d_mid),
            nn.ReLU(inplace=True),
        )
    def forward(self, p, pj, x, xj):
        # x: n, c 
        return self.mlp(x) 

class PointNet2(nn.Module):
    def __init__(self, d_in, d_out, config):
        super().__init__()
        self.expansion = 1 
        d_mid = d_out * self.expansion 
        self.mlp = nn.Sequential(
            nn.Conv1d(d_in+3, d_mid, 1),
            nn.BatchNorm1d(d_mid),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_mid, d_out, 1),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, p, pj, x, xj):
        l2 = torch.norm(pj, dim=-1, keepdim=True)
        pj = pj / (torch.max(l2, dim=1, keepdim=True)[0] + 1e-8)
        x = torch.cat([pj, xj], dim=-1) # (n, 3+c)
        x = x.permute(0, 2, 1).contiguous()
        x = self.mlp(x) 
        return x.max(2)[0]
        
class SimpleBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        func = config.convolution
        self.func = block_decider(func)(d_in, d_out, config)
        self.nsample = nsample
        self.level = level
    
    def forward(self, p, x, o):
        N, C = x.size()
        pj_xj, idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_index=True)  # (m,nsample, 3+c)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        x = self.func(p, pj, x, xj)
        return p, x, o

class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        func = config.convolution
        bottleneck_ratio = config.bottleneck_ratio
        self.level = level
        if bottleneck_ratio is None:
            self.reduction = self.expansion = nn.Identity()
            self.func = block_decider(func)(d_in, d_out, config)
        else:
            d_mid = d_in // bottleneck_ratio
            self.reduction = nn.Sequential(
                nn.Linear(d_in, d_mid),
                nn.BatchNorm1d(d_mid),
                nn.ReLU(inplace=True)
            )
            self.func = block_decider(func)(d_mid, d_mid, config)
            self.expansion = nn.Sequential(
                nn.Linear(d_mid, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True)
            )
        self.nsample = nsample
    
    def forward(self, p, x, o):
        N, C = x.size()
        identity = x
        x = self.reduction(x)
        pj_xj, idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_index=True)  # (m, nsample, 3+c)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        x = self.func(p, pj, x, xj)
        x = self.expansion(x)
        x = identity + x
        return p, x, o

class ResidualFusionBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        self.level = level
        self.nsample = nsample
        funcs = config.side_transform.split('/')
        if d_in != d_out:
            self.reduction = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.reduction = nn.Identity()

        self.funcs = nn.ModuleList()
        for func in funcs:
            self.funcs.append(block_decider(func)(d_out, d_out, config))
        if d_in != d_out:
            self.mlp_dim_align = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.mlp_dim_align = nn.Identity()
    
    def forward(self, p, x, o):
        N, C = x.size()
        identity = x
        x = self.reduction(x)
        pj_xj, idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_index=True)  # (m, nsample, 3+c)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        for f in self.funcs:
            x = f(p, pj, x, xj)
            xj = x[idx, :].view(N, self.nsample, -1)
        x = self.mlp_dim_align(identity) + x
        return p, x, o
        
class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_dense, d_out, nsample, stride, config, level=None):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_dense
        self.nsample = nsample
        self.d_out = d_out
        self.level=level

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse+ d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, p1,x1,o1, p2,x2,o2):
        '''
            pxo1: dense 
            pxo2: sparse  
        '''
        interpolated = pointops.interpolation(p2, p1, x2, o2, o1)
        x = self.mlp(torch.cat([x1, interpolated], dim=1))
        return p1, x, o1 

class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        self.d_in = d_in
        self.nsample = 16 
        self.stride = stride 
        self.mlp = nn.Sequential(
            nn.Linear(d_in+3, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, p, x, o):
        identity = x

        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i-1].item()) // self.stride
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o)
        idx = pointops.furthestsampling(p, o, n_o)  # (m)
        n_p = p[idx.long(), :]  # (m, 3)
        
        pj_xj, idx = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True, return_index=True)  # (m, 3+c, nsample)
        pj, xj = pj_xj[:, :, :3], pj_xj[:, :, 3:]
        pj = pj / (torch.max(torch.norm(pj, dim=-1, keepdim=True), dim=1, keepdim=True)[0] + 1e-8)
        pj_xj = torch.cat([pj, xj], dim=-1)
        x = self.mlp(pj_xj.max(1)[0])
        return n_p, x, n_o

class AdaptiveLaplacian(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        self.nsample = nsample
        self.level = level
        self.pre_trans = nn.Linear(d_in, d_in)
        self.activation = nn.ReLU(inplace=True)
        self.varphi = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(inplace=True)
        )   

    def forward(self, p, u, o):
        # p, u, o = pxo # (n,3), (n, c), (b)
        N, C = u.size()
        u_t = u
        u = self.pre_trans(u)
        u_n, idx = pointops.queryandgroup(self.nsample, p, p, u, None, o, o, use_xyz=False, return_index=True) # (n,nsample,c)
        Lap = self.activation(u_n - u[:, None, :]).mean(1)
        
        u_tt = self.varphi(Lap)

        return p, u_tt, o

class EdgeConditioning(nn.Module):
    def __init__(self, d_in, d_in_edge, d_out, nsample, stride, config, level=None):
        super().__init__()
        self.concat_fusion = nn.Sequential(
            nn.Linear(d_in + d_in_edge, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(inplace=True)
        )
        self.conditioning = lambda x,y: x + self.concat_fusion(torch.cat([x,y], dim=1)) 
    def forward(self, x, edge):
        return self.conditioning(x, edge) 


