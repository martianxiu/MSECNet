import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F

import os
import sys
import numpy as np
from blocks import *

class MSECNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config.d_in_initial
        d_out = config.d_out_initial 
        n_cls = config.num_classes
        nsample = config.nsample
        self.nsample_side = config.nsample_side 
        self.nsample_interp = config.nsample_interp
        stride_list = config.strides
        stride = 1
        stride_idx = 0
        d_prev = d_in 
        level = 0
        self.n_scale = config.n_scale

        # construct encoder 
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        self.side_output_dim = [] 
        layer_ind_major = 0
        layer_ind_sub = -1
        for block_i, block_name in enumerate(config.architecture):
            layer_ind_sub += 1
            # Detect change to next layer for skip connection
            if np.any([tmp in block_name for tmp in ['strided', 'downsample']]):
                self.encoder_skip_dims.append(d_prev)
                self.encoder_skips.append(block_i)
                layer_ind_major += 1
                layer_ind_sub = 0 
                level += 1

            # Detect upsampling block to stop
            if 'upsample' in block_name:
                break

            # update feature dim            
            d_in = d_prev 
            # if subsample
            if 'strided' in block_name or 'downsample' in block_name:
                self.side_output_dim.append(d_out)
                stride = stride_list[stride_idx]
                stride_idx += 1
                d_out *= 2
            else:
                stride = 1
            
            # stack modules
            nsample = config.nsample

            layer_ind = f'e_{str(layer_ind_major)}_{str(layer_ind_sub)}'
            self.encoder_blocks.append(
                block_decider(block_name)(
                    d_in, d_out, nsample, stride, config, layer_ind 
                )
            )

            d_prev = d_out

        self.side_output_dim.append(d_out) # for the last scale 

        # construct decoder 
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = []

        # Find first upsampling block
        start_i = 0
        for block_i, block_name in enumerate(config.architecture):
            if 'upsample' in block_name:
                start_i = block_i
                break

        # Loop over consecutive blocks
        layer_ind_major = 0
        layer_ind_sub = -1 
        for block_i, block_name in enumerate(config.architecture[start_i:]):
            layer_ind_sub += 1

            d_in = d_out

            # detect the upsample layer
            if 'upsample' in block_name:
                layer_ind_major += 1
                layer_ind_sub = 0
                level -= 1

                self.decoder_upsample.append(block_i)
                
                # if upsample, out_dim / 2 
                d_out = max(d_out // 2, config.decoder_out_dim)                

                nsample = config.nsample

                layer_ind = f'd_{str(layer_ind_major)}_{str(layer_ind_sub)}'
                self.decoder_blocks.append(
                    block_decider(block_name)(
                        [d_in, self.encoder_skip_dims.pop()], 
                        d_out, nsample, stride, config, layer_ind 
                    )
                )
            else:
                # if not upsample, then dim remain same
                nsample = config.nsample

                layer_ind = f'd_{str(layer_ind_major)}_{str(layer_ind_sub)}'
                self.decoder_blocks.append(
                    block_decider(block_name)(
                        d_in, d_in, nsample, stride, config, layer_ind 
                    )
                )
        
        d_fusion = config.d_fusion 
        print(f'Number of scales: {len(self.side_output_dim)}, use {self.n_scale}, the fused channel dim: {d_fusion}')
        self.interp_weight_type = config.interp_weight_type
        
        self.ms_fusion = block_decider(config.side_transform_block)(sum(self.side_output_dim[:self.n_scale]), d_fusion, self.nsample_side, None, config, 'nn_transform')
        self.edge_transfrom = block_decider(config.edge_detector)(d_fusion, d_fusion, self.nsample_side, None, config, 'edge_transform')
        self.ee = block_decider('edge_conditioning')(d_out, d_fusion, d_out, self.nsample_side, None, config, 'edge_conditioning')
         
        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_out, d_out), 
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_out, n_cls)
        )        
    

    def forward(self, p, x, o, save_path=None):
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []

        side_output = []
        # encoder
        for block_i, block in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                p_from_encoder.append(p) 
                x_from_encoder.append(x)
                o_from_encoder.append(o)
                side_output.append([p, x, o])
            p, x, o = block(p, x, o)
        side_output.append([p,x,o])

        # decoder
        for block_i, block in enumerate(self.decoder_blocks):
            if block_i in self.decoder_upsample:
                p_dense = p_from_encoder.pop()
                x_dense = x_from_encoder.pop()
                o_dense = o_from_encoder.pop()
                p, x, o = block(p_dense, x_dense, o_dense, p, x, o) 
            else:
                p, x, o = block(p, x, o) 
            
        # MSEC branch 
        p_dense, x_dense, o_dense = side_output[0] # the first one is the most dense 
        ms_feat = x_dense # n, c
        for i in range(1, len(side_output[:self.n_scale])):
            p_sp, x_sp, o_sp = side_output[i]
            interpolated = pointops.interpolation_flexible(p_sp, p_dense, x_sp, o_sp, o_dense, k=self.nsample_interp, weight_type=self.interp_weight_type) 
            ms_feat = torch.cat([ms_feat, interpolated], dim=1) # n,c 
        
        ms_feat_new = self.ms_fusion(p_dense, ms_feat, o_dense)[1]
        n,c = ms_feat_new.shape
        ms_edge = self.edge_transfrom(p_dense, ms_feat_new, o_dense)[1]
        
        x_out = self.ee(x, ms_edge)

        if save_path is not None:
            self.save_features(save_path, p_dense, ms_feat, ms_feat_new, ms_edge, x, x_out)

        # classification     
        x_out = self.classifier(x_out)
        return x_out

