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

# class for discriminator of composition GAN
class Vid2VidModelD(BaseModel):
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
        self.tD = opt.n_frames_D  
        self.output_nc = opt.output_nc_3        

        # define networks        
        # single image discriminator

        netD_input_nc_2 = opt.input_nc_T_3 + opt.input_nc_S_3 + 3 + opt.output_nc_3
        netD_input_nc_1 = netD_input_nc_2 + opt.output_nc_3
        netD_input_nc_f = (3+1+3) + (3+1) + opt.output_nc_3
               
        self.netD = networks.define_D(netD_input_nc_1, opt.ndf, opt.n_layers_D, opt.norm,
                                      opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids)
        self.netD_FG = networks.define_D(netD_input_nc_2, opt.ndf, opt.n_layers_D, opt.norm,
                                      opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids)

        if opt.add_face_disc:            
            self.netD_f = networks.define_D(netD_input_nc_f, opt.ndf, opt.n_layers_D, opt.norm,
                                            max(1, opt.num_D - 2), not opt.no_ganFeat, gpu_ids=self.gpu_ids)
                    
        # temporal discriminator
        netD_input_nc = opt.output_nc_3 * opt.n_frames_D + 2 * (opt.n_frames_D-1)        
        for s in range(opt.n_scales_temporal):
            setattr(self, 'netD_T'+str(s), networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                    opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids))        

        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if opt.continue_train or opt.load_pretrain:          
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)            
            self.load_network(self.netD_FG, 'D_FG', opt.which_epoch, opt.load_pretrain)            
            for s in range(opt.n_scales_temporal):
                self.load_network(getattr(self, 'netD_T'+str(s)), 'D_T'+str(s), opt.which_epoch, opt.load_pretrain)
            if opt.add_face_disc:
                self.load_network(self.netD_f, 'D_f', opt.which_epoch, opt.load_pretrain)
           
        # set loss functions and optimizers          
        self.old_lr = opt.lr
        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor)   
        self.criterionFlow = networks.MaskedL1Loss()
        self.criterionWarp = networks.MaskedL1Loss()
        self.criterionFeat = torch.nn.L1Loss()
        self.L1Loss = torch.nn.L1Loss()
        if not opt.no_vgg:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids[0])

        self.loss_names = ['G_SI_VGG', 'G_SI_GAN', 'G_SI_GAN_Feat',                            
                           'D_SI_real', 'D_SI_fake',
                           'G_SFG_res_L1', 'G_SFG_res_VGG', 'G_SFG_GAN', 'G_SFG_GAN_Feat',
                           'D_SFG_real', 'D_SFG_fake',
                           'G_Warp', 'F_Flow', 'F_Warp', 'W']                
        self.loss_names_T = ['G_T_GAN', 'G_T_GAN_Feat', 'D_T_real', 'D_T_fake', 'G_T_Warp']     
        if opt.add_face_disc:
            self.loss_names += ['G_f_GAN', 'G_f_GAN_Feat', 'D_f_real', 'D_f_fake']

        # initialize optimizers D and D_T                                            
        params = list(self.netD.parameters())
        params += list(self.netD_FG.parameters())
        if opt.add_face_disc:
            params += list(self.netD_f.parameters())
        if opt.TTUR:                
            beta1, beta2 = 0, 0.9
            lr = opt.lr * 2
        else:
            beta1, beta2 = opt.beta1, 0.999
            lr = opt.lr
        self.optimizer_D = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))        

        for s in range(opt.n_scales_temporal):            
            params = list(getattr(self, 'netD_T'+str(s)).parameters())          
            optimizer_D_T = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))            
            setattr(self, 'optimizer_D_T'+str(s), optimizer_D_T)    

    def forward(self, scale_T, tensors_list, dummy_bs=0):
        lambda_feat = self.opt.lambda_feat
        lambda_F = self.opt.lambda_F
        lambda_T = self.opt.lambda_T
        scale_S = self.opt.n_scales_spatial
        tD = self.opt.n_frames_D
        if tensors_list[0].get_device() == self.gpu_ids[0]:
            tensors_list = util.remove_dummy_from_tensor(tensors_list, dummy_bs)
            if tensors_list[0].size(0) == 0:                
                return [self.Tensor(1, 1).fill_(0)] * (len(self.loss_names_T) if scale_T > 0 else len(self.loss_names))
        
        if scale_T > 0:
            real_SI, fake_SI, flow_ref, conf_ref = tensors_list
            _, _, _, self.height, self.width = real_SI.size()
            loss_D_T_real, loss_D_T_fake, loss_G_T_GAN, loss_G_T_GAN_Feat = self.compute_loss_D_T(real_SI, fake_SI, 
                flow_ref/20, conf_ref, scale_T-1)            
            loss_G_T_Warp = torch.zeros_like(loss_G_T_GAN)

            loss_list = [loss_G_T_GAN, loss_G_T_GAN_Feat, loss_D_T_real, loss_D_T_fake, loss_G_T_Warp]
            loss_list = [loss.view(-1, 1) for loss in loss_list]
            return loss_list            
        
        real_SI, real_SFG_full, fake_SI, fake_SI_raw, fake_SFG_full, fake_SFG_res, real_input_T, real_input_S, real_input_SFG, real_input_BG \
        , real_input_BG_flag, real_SI_prev, fake_SI_prev, real_SFG_full_prev, flow, weight, flow_ref, conf_ref = tensors_list
        real_input_T = real_input_T.expand(-1, real_input_S.size()[1], -1, -1, -1)
        real_input_S_full = torch.cat([real_input_S, real_input_SFG], dim=2)
        real_TLO_f = real_input_T[:, :, 3+2:3+3]
        real_TP_f = real_input_T[:, :, 0:3]
        real_TFG_f = real_input_T[:, :, -3:]
        real_input_T_f = torch.cat([real_TP_f, real_TLO_f], dim=2)
        real_input_T_f = torch.cat([real_input_T_f, real_TFG_f], dim=2)
        real_SLO_f = real_input_S[:, :, self.opt.label_nc_2+1-self.opt.label_nc_3:self.opt.label_nc_2+2-self.opt.label_nc_3]
        real_SP_f = real_input_S[:, :, -3-self.opt.label_nc_3:-self.opt.label_nc_3]
        real_input_S_f = torch.cat([real_SP_f, real_SLO_f], dim=2)

        self.bs = real_SI.size()[0]
        for b in range(self.bs):
            if not real_input_BG_flag[b]:
                real_SI[b] = real_SFG_full[b]
                real_SI_prev[b] = real_SFG_full_prev[b]
                #fake_SI[b] = fake_SFG_full[b]
                #fake_SI_raw[b] = fake_SFG_full[b]
                #flow[b] = flow_ref[b]

        real_SI, real_SFG_full, fake_SI, fake_SI_raw, fake_SFG_full, fake_SFG_res, real_input_T, real_input_S_full, real_input_BG \
        , real_SI_prev, fake_SI_prev, flow, weight, flow_ref, conf_ref, real_input_T_f, real_input_S_f \
        = reshape([real_SI, real_SFG_full, fake_SI, fake_SI_raw, fake_SFG_full, fake_SFG_res, real_input_T, real_input_S_full, real_input_BG \
                  , real_SI_prev, fake_SI_prev, flow, weight, flow_ref, conf_ref, real_input_T_f, real_input_S_f])
        _, _, self.height, self.width = real_SI.size()

        ################### Flow loss #################
        if flow is not None:
            # similar to flownet flow        
            loss_F_Flow = self.criterionFlow(flow, flow_ref, conf_ref) * lambda_F / (2 ** (scale_S-1))        
            # warped prev image should be close to current image            
            real_SI_warp = self.resample(real_SI_prev, flow)                
            loss_F_Warp = self.criterionFlow(real_SI_warp, real_SI, conf_ref) * lambda_T
            
            ################## weight loss ##################
            loss_W = torch.zeros_like(weight)
            if self.opt.no_first_img:
                dummy0 = torch.zeros_like(weight)
                loss_W = self.criterionFlow(weight, dummy0, conf_ref)
        else:
            loss_F_Flow = loss_F_Warp = loss_W = torch.zeros_like(conf_ref)

        #################### fake_B loss ####################        
        ### VGG + GAN loss 

        loss_G_SI_VGG = (self.criterionVGG(fake_SI, real_SI) * lambda_feat) if not self.opt.no_vgg else torch.zeros_like(loss_W)
        real_input_1 = torch.cat([real_input_T, real_input_S_full], dim=1)
        real_input_1 = torch.cat([real_input_1, real_input_BG], dim=1)
        loss_D_SI_real, loss_D_SI_fake, loss_G_SI_GAN, loss_G_SI_GAN_Feat = self.compute_loss_D(self.netD, real_input_1, real_SI, fake_SI)
        ### Warp loss
        fake_SI_warp_ref = self.resample(fake_SI_prev, flow_ref)
        loss_G_Warp = self.criterionWarp(fake_SI, fake_SI_warp_ref.detach(), conf_ref) * lambda_T

        if fake_SI_raw is not None:
            if not self.opt.no_vgg:
                loss_G_SI_VGG += self.criterionVGG(fake_SI_raw, real_SI) * lambda_feat        
            l_D_SI_real, l_D_SI_fake, l_G_SI_GAN, l_G_SI_GAN_Feat = self.compute_loss_D(self.netD, real_input_1, real_SI, fake_SI_raw)        
            loss_G_SI_GAN += l_G_SI_GAN; loss_G_SI_GAN_Feat += l_G_SI_GAN_Feat
            loss_D_SI_real += l_D_SI_real; loss_D_SI_fake += l_D_SI_fake

        #loss_G_SFG_VGG = (self.criterionVGG(fake_SFG_full, real_SFG_full) * lambda_feat) if not self.opt.no_vgg else torch.zeros_like(loss_W)
        loss_G_SFG_res_L1 = self.L1Loss(fake_SFG_res, real_SFG_full) * lambda_feat
        loss_G_SFG_res_VGG = (self.criterionVGG(fake_SFG_res, real_SFG_full) * lambda_feat) if not self.opt.no_vgg else torch.zeros_like(loss_W)
        real_input_2 = torch.cat([real_input_T, real_input_S_full], dim=1)
        loss_D_SFG_real, loss_D_SFG_fake, loss_G_SFG_GAN, loss_G_SFG_GAN_Feat = self.compute_loss_D(self.netD_FG, real_input_2, real_SFG_full, fake_SFG_full)

        if self.opt.add_face_disc:
            face_weight = 2
            ys_T, ye_T, xs_T, xe_T = self.get_face_region(real_input_T_f[:, 3:4])
            ys_S, ye_S, xs_S, xe_S = self.get_face_region(real_input_S_f[:, 3:4])
            if ys_T is not None and ys_S is not None:                
                real_input_f = torch.cat([real_input_T_f[:,:,ys_T:ye_T,xs_T:xe_T], real_input_S_f[:,:,ys_S:ye_S,xs_S:xe_S]], dim=1)
                loss_D_f_real, loss_D_f_fake, loss_G_f_GAN, loss_G_f_GAN_Feat = self.compute_loss_D(self.netD_f,
                    real_input_f, real_SFG_full[:,:,ys_S:ye_S,xs_S:xe_S], fake_SFG_full[:,:,ys_S:ye_S,xs_S:xe_S])  
                loss_G_f_GAN *= face_weight  
                loss_G_f_GAN_Feat *= face_weight                  
            else:
                loss_D_f_real = loss_D_f_fake = loss_G_f_GAN = loss_G_f_GAN_Feat = torch.zeros_like(loss_D_SFG_real)

        loss_list = [loss_G_SI_VGG, loss_G_SI_GAN, loss_G_SI_GAN_Feat,
                     loss_D_SI_real, loss_D_SI_fake, 
                     loss_G_SFG_res_L1, loss_G_SFG_res_VGG, loss_G_SFG_GAN, loss_G_SFG_GAN_Feat,
                     loss_D_SFG_real, loss_D_SFG_fake,
                     loss_G_Warp, loss_F_Flow, loss_F_Warp, loss_W]
        if self.opt.add_face_disc:
            loss_list += [loss_G_f_GAN, loss_G_f_GAN_Feat, loss_D_f_real, loss_D_f_fake]   
        loss_list = [loss.view(-1, 1) for loss in loss_list]           
        return loss_list

    def compute_loss_D(self, netD, real_A, real_B, fake_B):        
        real_AB = torch.cat((real_A, real_B), dim=1)
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        pred_real = netD.forward(real_AB)
        pred_fake = netD.forward(fake_AB.detach())
        loss_D_real = self.criterionGAN(pred_real, True) 
        loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_fake = netD.forward(fake_AB)                       
        loss_G_GAN, loss_G_GAN_Feat = self.GAN_and_FM_loss(pred_real, pred_fake)      

        return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat      

    def compute_loss_D_T(self, real_B, fake_B, flow_ref, conf_ref, scale_T):         
        netD_T = getattr(self, 'netD_T'+str(scale_T))
        real_B = real_B.view(-1, self.output_nc * self.tD, self.height, self.width)
        fake_B = fake_B.view(-1, self.output_nc * self.tD, self.height, self.width)        
        if flow_ref is not None:
            flow_ref = flow_ref.view(-1, 2 * (self.tD-1), self.height, self.width)                        
            real_B = torch.cat([real_B, flow_ref], dim=1)
            fake_B = torch.cat([fake_B, flow_ref], dim=1)
        pred_real = netD_T.forward(real_B)
        pred_fake = netD_T.forward(fake_B.detach())
        loss_D_T_real = self.criterionGAN(pred_real, True)            
        loss_D_T_fake = self.criterionGAN(pred_fake, False)        

        pred_fake = netD_T.forward(fake_B)                               
        loss_G_T_GAN, loss_G_T_GAN_Feat = self.GAN_and_FM_loss(pred_real, pred_fake)

        return loss_D_T_real, loss_D_T_fake, loss_G_T_GAN, loss_G_T_GAN_Feat

    def GAN_and_FM_loss(self, pred_real, pred_fake):
        ### GAN loss            
        loss_G_GAN = self.criterionGAN(pred_fake, True)                             

        # GAN feature matching loss
        loss_G_GAN_Feat = torch.zeros_like(loss_G_GAN)
        if not self.opt.no_ganFeat:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(min(len(pred_fake), self.opt.num_D)):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        return loss_G_GAN, loss_G_GAN_Feat

    def get_face_region(self, real_A):
        _, _, h, w = real_A.size()
        face = (real_A[:,0] != 0).nonzero()
        #face = (((real_A[:,9] == 0.6) | (real_A[:,9] == 0.2)) & (real_A[:,10] == 0) & (real_A[:,11] == 0.6)).nonzero()
        if face.size()[0]:
            y, x = face[:,1], face[:,2]
            ys, ye, xs, xe = y.min().item(), y.max().item(), x.min().item(), x.max().item()
            yc, ylen = int(ys+ye)//2, 32
            xc, xlen = int(xs+xe)//2, 32
            yc = max(ylen//2, min(h-1 - ylen//2, yc))
            xc = max(xlen//2, min(w-1 - xlen//2, xc))
            ys, ye, xs, xe = yc - ylen//2, yc + ylen//2, xc - xlen//2, xc + xlen//2
            return ys, ye, xs, xe
        return None, None, None, None

    def get_all_skipped_frames(self, frames_all, real_B, fake_B, flow_ref, conf_ref, t_scales, tD, n_frames_load, i, flowNet):
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all = frames_all
        if t_scales > 0:
            if self.opt.sparse_D:          
                real_B_all, real_B_skipped = get_skipped_frames_sparse(real_B_all, real_B, t_scales, tD, n_frames_load, i)
                fake_B_all, fake_B_skipped = get_skipped_frames_sparse(fake_B_all, fake_B, t_scales, tD, n_frames_load, i)
                flow_ref_all, flow_ref_skipped = get_skipped_frames_sparse(flow_ref_all, flow_ref, t_scales, tD, n_frames_load, i, is_flow=True)
                conf_ref_all, conf_ref_skipped = get_skipped_frames_sparse(conf_ref_all, conf_ref, t_scales, tD, n_frames_load, i, is_flow=True)
            else:
                real_B_all, real_B_skipped = get_skipped_frames(real_B_all, real_B, t_scales, tD)
                fake_B_all, fake_B_skipped = get_skipped_frames(fake_B_all, fake_B, t_scales, tD)                
                flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped = get_skipped_flows(flowNet, 
                    flow_ref_all, conf_ref_all, real_B_skipped, flow_ref, conf_ref, t_scales, tD)    
        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all
        frames_skipped = real_B_skipped, fake_B_skipped, flow_ref_skipped, conf_ref_skipped
        return frames_all, frames_skipped

    def get_losses(self, loss_dict, loss_dict_T, t_scales):
        loss_D = (loss_dict['D_SI_fake'] + loss_dict['D_SI_real'] + loss_dict['D_SFG_fake'] + loss_dict['D_SFG_real']) * 0.5
        loss_G = loss_dict['G_SI_GAN'] + loss_dict['G_SI_GAN_Feat'] + loss_dict['G_SI_VGG'] + loss_dict['G_SFG_GAN'] + loss_dict['G_SFG_GAN_Feat'] + loss_dict['G_SFG_res_L1'] + loss_dict['G_SFG_res_VGG']
        loss_G += loss_dict['G_Warp'] + loss_dict['F_Flow'] + loss_dict['F_Warp'] + loss_dict['W']
        if self.opt.add_face_disc:
            loss_G += loss_dict['G_f_GAN'] + loss_dict['G_f_GAN_Feat'] 
            loss_D += (loss_dict['D_f_fake'] + loss_dict['D_f_real']) * 0.5
              
        # collect temporal losses
        loss_D_T = []           
        t_scales_act = min(t_scales, len(loss_dict_T))            
        for s in range(t_scales_act):
            loss_G += loss_dict_T[s]['G_T_GAN'] + loss_dict_T[s]['G_T_GAN_Feat'] + loss_dict_T[s]['G_T_Warp']                
            loss_D_T.append((loss_dict_T[s]['D_T_fake'] + loss_dict_T[s]['D_T_real']) * 0.5)

        return loss_G, loss_D, loss_D_T, t_scales_act

    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)         
        self.save_network(self.netD_FG, 'D_FG', label, self.gpu_ids)         
        for s in range(self.opt.n_scales_temporal):
            self.save_network(getattr(self, 'netD_T'+str(s)), 'D_T'+str(s), label, self.gpu_ids)   
        if self.opt.add_face_disc:
            self.save_network(self.netD_f, 'D_f', label, self.gpu_ids)

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

def reshape(tensors):
    if tensors is None: return None
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]    
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)