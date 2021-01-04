### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks

# class for losses including FTC loss, VGG loss, TVL1 loss and other auxiliary losses of our C2F-FWN
class ClothWarperLoss(BaseModel):
    def name(self):
        return 'Vid2VidModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)        
        gpu_split_id = opt.n_gpus_gen
        if opt.batchSize == 1:
            gpu_split_id += 1
        self.gpu_ids = ([opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:]) if opt.n_gpus_gen != len(opt.gpu_ids) else opt.gpu_ids
        if not opt.debug:
            torch.backends.cudnn.benchmark = True    
           
        # set loss functions and optimizers          
        self.old_lr = opt.lr
        # define loss functions
        self.MaskedL1Loss = networks.MaskedL1Loss()
        self.L1Loss = torch.nn.L1Loss()
        self.VGGLoss = networks.VGGLoss(self.gpu_ids[0])
        self.TVLoss = networks.TVLoss()

        self.loss_names = ['FG_L1', 'FG_VGG', 'FG_TPS_L1', 'FG_TPS_VGG', 'LO_L1', 'LO_TPS_L1', 'Flow_O', 'Flow_TVL1']     
        self.loss_names_T = ['Flow_O_T']

    def forward(self, scale_T, tensors_list, is_first_frame, dummy_bs=0):
        lambda_feat = self.opt.lambda_feat
        lambda_structure = self.opt.lambda_structure
        lambda_flow = self.opt.lambda_flow
        lambda_smooth = self.opt.lambda_smooth
        if tensors_list[0].get_device() == self.gpu_ids[0]:
            tensors_list = util.remove_dummy_from_tensor(tensors_list, dummy_bs)
            if tensors_list[0].size(0) == 0:                
                return [self.Tensor(1, 1).fill_(0)] * len(self.loss_names)

        if scale_T > 0:
            real_sfg_full, flow_total, flow_ref, conf_ref, real_SLO = tensors_list
            _, _, _, self.height, self.width = real_sfg_full.size()
    
            real_SMask = real_SLO[:,1,1:].sum(dim=1, keepdim=True)

            flow_total_warp = self.resample(flow_total[:,0], flow_ref[:,0]) + flow_ref[:,0]
            loss_Flow_O = self.MaskedL1Loss(flow_total_warp, flow_total[:,1], real_SMask*conf_ref[:,0])
            loss_Flow_O = loss_Flow_O * 5.0

            loss_list = [loss_Flow_O]
            loss_list = [loss.view(-1, 1) for loss in loss_list]

            return loss_list, [flow_total_warp*real_SMask*conf_ref[:,0], flow_total[:,0], flow_total[:,1]*real_SMask*conf_ref[:,0]]

        real_SFG, real_SLO, fg_tps, fg_dense, lo_tps, lo_dense, flow_tps, flow_dense, flow_total, flow_total_prev, flow_optical_ref, conf_optical_ref = tensors_list
        _, _, self.height, self.width = real_SFG.size()

        real_SMask = real_SLO[:,1:].sum(dim=1, keepdim=True)

        ################## Image loss #################
        if not self.opt.tps_only:
            loss_FG_L1 = self.L1Loss(fg_dense, real_SFG) * 10
            loss_FG_VGG = self.VGGLoss(fg_dense, real_SFG)
        else:
            loss_FG_L1 = None
            loss_FG_VGG = None
        loss_FG_TPS_L1 = self.L1Loss(fg_tps, real_SFG) * 10
        loss_FG_TPS_VGG = self.VGGLoss(fg_tps, real_SFG)
        if not self.opt.tps_only:
            loss_LO_L1 = self.L1Loss(lo_dense, real_SLO) * 20
            #loss_LO_L1 = self.L1Loss(lo_dense[:,0:-2], real_SLO[:,0:-2]) * 10
            #loss_LO_L1 = loss_LO_L1 + self.L1Loss(lo_dense[:,-2:], real_SLO[:,-2:]) * 50
        else:
            loss_LO_L1 = None
        loss_LO_TPS_L1 = self.L1Loss(lo_tps, real_SLO) * 10
        #loss_LO_TPS_L1 = self.L1Loss(lo_tps[:,0:-2], real_SLO[:,0:-2]) * 10
        #loss_LO_TPS_L1 = loss_LO_TPS_L1 + self.L1Loss(lo_tps[:,-2:], real_SLO[:,-2:]) * 50

        ################### Flow loss #################
        if not self.opt.tps_only:
            flow_total_warp = self.resample(flow_total_prev, flow_optical_ref[-flow_total_prev.size()[0]:]) + flow_optical_ref[-flow_total_prev.size()[0]:]
        #print(flow_total_warp.size(), flow_total.size(), real_SMask.size(), conf_optical_ref.size())
            loss_Flow_O = self.MaskedL1Loss(flow_total_warp, flow_total[-flow_total_prev.size()[0]:], real_SMask[-flow_total_prev.size()[0]:]*conf_optical_ref[-flow_total_prev.size()[0]:])
            loss_Flow_O = loss_Flow_O * 5.0
        else:
            flow_total_warp = torch.zeros_like(flow_total_prev)
            loss_Flow_O = None
        if not self.opt.tps_only:
            loss_Flow_TVL1 = self.TVLoss(flow_total) * 0.5
        else:
            loss_Flow_TVL1 = None

        loss_list = [loss_FG_L1, loss_FG_VGG, loss_FG_TPS_L1, loss_FG_TPS_VGG, loss_LO_L1, loss_LO_TPS_L1, loss_Flow_O, loss_Flow_TVL1]
        loss_list = [loss.view(-1, 1) if loss is not None else None for loss in loss_list]           
        return loss_list, [flow_total_warp*real_SMask[-flow_total_prev.size()[0]:]*conf_optical_ref[-flow_total_prev.size()[0]:], flow_total_prev, flow_total[-flow_total_prev.size()[0]:]*real_SMask[-flow_total_prev.size()[0]:]*conf_optical_ref[-flow_total_prev.size()[0]:]]

    def get_all_skipped_frames(self, frames_all, real_B, fake_B, flow_ref, conf_ref, real_SLO, t_scales, tD, n_frames_load, i, flowNet):
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all, real_SLO_all = frames_all
        if t_scales > 0:
            if self.opt.sparse_D:          
                real_B_all, real_B_skipped = get_skipped_frames_sparse(real_B_all, real_B, t_scales, tD, n_frames_load, i)
                fake_B_all, fake_B_skipped = get_skipped_frames_sparse(fake_B_all, fake_B, t_scales, tD, n_frames_load, i)
                flow_ref_all, flow_ref_skipped = get_skipped_frames_sparse(flow_ref_all, flow_ref, t_scales, tD, n_frames_load, i, is_flow=True)
                conf_ref_all, conf_ref_skipped = get_skipped_frames_sparse(conf_ref_all, conf_ref, t_scales, tD, n_frames_load, i, is_flow=True)
            else:
                real_B_all, real_B_skipped = get_skipped_frames(real_B_all, real_B, t_scales, tD)
                fake_B_all, fake_B_skipped = get_skipped_frames(fake_B_all, fake_B, t_scales, tD)                
                real_SLO_all, real_SLO_skipped = get_skipped_frames(real_SLO_all, real_SLO, t_scales, tD)                
                flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped = get_skipped_flows(flowNet, 
                    flow_ref_all, conf_ref_all, real_B_skipped, flow_ref, conf_ref, t_scales, tD)    
        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all, real_SLO_all
        frames_skipped = real_B_skipped, fake_B_skipped, flow_ref_skipped, conf_ref_skipped, real_SLO_skipped
        return frames_all, frames_skipped

    def get_losses(self, loss_dict, loss_dict_T, t_scales):
        loss = loss_dict['FG_L1'] + loss_dict['FG_VGG'] + loss_dict['FG_TPS_L1'] + loss_dict['FG_TPS_VGG'] + loss_dict['LO_L1'] + loss_dict['LO_TPS_L1'] + loss_dict['Flow_O'] + loss_dict['Flow_TVL1']

        # collect temporal losses
        t_scales_act = min(t_scales, len(loss_dict_T))            
        for s in range(t_scales_act):
            if loss_dict_T[s] is not None:
                loss += loss_dict_T[s]['Flow_O_T'] * 0.5**(s+1)

        return loss, t_scales_act

# get temporally subsampled frames for real/fake sequences
def get_skipped_frames(B_all, B, t_scales, tD):
    B_all = torch.cat([B_all.detach(), B], dim=1) if B_all is not None else B
    B_skipped = [None] * t_scales
    for s in range(t_scales):
        tDs = tD ** s        # number of skipped frames between neighboring frames (e.g. 1, 3, 9, ...)
        span = tDs * (tD-1)  # number of frames the final triplet frames span before skipping (e.g., 2, 6, 18, ...)
        n_groups = min(B_all.size()[1] - span, B.size()[1])
        if n_groups > 0:
            for t in range(0, n_groups, tD):
                skip = B_all[:, (-span-t-1):-t:tDs].contiguous() if t != 0 else B_all[:, -span-1::tDs].contiguous()                
                B_skipped[s] = torch.cat([B_skipped[s], skip]) if B_skipped[s] is not None else skip             
    max_prev_frames = tD ** (t_scales-1) * (tD-1)
    if B_all.size()[1] > max_prev_frames:
        B_all = B_all[:, -max_prev_frames:]
    return B_all, B_skipped

# get temporally subsampled frames for flows
def get_skipped_flows(flowNet, flow_ref_all, conf_ref_all, real_B, flow_ref, conf_ref, t_scales, tD):  
    flow_ref_skipped, conf_ref_skipped = [None] * t_scales, [None] * t_scales  
    flow_ref_all, flow = get_skipped_frames(flow_ref_all, flow_ref, 1, tD)
    conf_ref_all, conf = get_skipped_frames(conf_ref_all, conf_ref, 1, tD)
    if flow[0] is not None:
        flow_ref_skipped[0], conf_ref_skipped[0] = flow[0][:,1:], conf[0][:,1:]

    for s in range(1, t_scales):        
        if real_B[s] is not None and real_B[s].size()[1] == tD:            
            flow_ref_skipped[s], conf_ref_skipped[s] = flowNet(real_B[s][:,1:], real_B[s][:,:-1])
    return flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped

def get_skipped_frames_sparse(B_all, B, t_scales, tD, n_frames_load, i, is_flow=False):
    B_skipped = [None] * t_scales
    _, _, ch, h, w = B.size()
    for s in range(t_scales):
        t_len = B_all[s].size()[1] if B_all[s] is not None else 0
        if t_len > 0 and (t_len % tD) == 0:
            B_all[s] = B_all[s][:, (-tD+1):] # get rid of unnecessary past frames        

        if s == 0:
            B_all[0] = torch.cat([B_all[0].detach(), B], dim=1) if B_all[0] is not None else B
        else:
            tDs = tD ** s
            idx_start = 0 if i == 0 else tDs - ((i-1) % tDs + 1)            
            if idx_start < n_frames_load:
                tmp = B[:, idx_start::tDs].contiguous()
                B_all[s] = torch.cat([B_all[s].detach(), tmp], dim=1) if B_all[s] is not None else tmp

        t_len = B_all[s].size()[1] if B_all[s] is not None else 0
        if t_len >= tD:            
            B_all[s] = B_all[s][:, (t_len % tD):]
            B_skipped[s] = B_all[s].view(-1, tD, ch, h, w)
            if is_flow:
                B_skipped[s] = B_skipped[s][:, 1:]

    return B_all, B_skipped