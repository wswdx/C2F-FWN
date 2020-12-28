### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_full
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import flowiz as fz
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
Parser, ClothWarper, Composer = create_model_full(opt)
visualizer = Visualizer(opt)

save_dir_input_1 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage1'))
save_dir_input_2 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage2'))
save_dir_input_3 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage3'))
save_dir_output_1 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_stage1'))
save_dir_output_2 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_stage2'))
save_dir_output_3 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_stage3'))
save_dir_output_4 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_SI'))
save_dir_output_5 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_SI_raw'))
save_dir_ref = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_source_ref'))

print('Doing %d frames' % len(dataset))

SPose_cloth = torch.zeros(1, 3, 3, 256, 192).cuda(opt.gpu_ids[0])
SParsing_cloth = torch.zeros(1, 3, 1, 256, 192).cuda(opt.gpu_ids[0])
SParsing = torch.zeros(1, 3, 1, 256, 192).cuda(opt.gpu_ids[0])

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break    
    if data['change_seq']:
        Parser.fake_slo_prev = None
        ClothWarper.flow_total_prev = None
        Composer.fake_SI_prev = None

    _, _, height, width = data['SI'].size()
    TParsing = Variable(data['TParsing']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    TParsing_uncloth = Variable(data['TParsing_uncloth']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    TFG_uncloth = Variable(data['TFG_uncloth']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    TParsing_cloth = Variable(data['TParsing_cloth']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    TFG_cloth = Variable(data['TFG_cloth']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    SPose = Variable(data['SPose']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    SParsing_ref = Variable(data['SParsing']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    SI_ref = Variable(data['SI']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    BG = Variable(data['BG']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])

    if data['change_seq'] or i == 0:
        for t in range(3):
            SParsing_sm_t, input_T_stage1, input_S_stage1 = Parser.inference(TParsing, SPose[:,t:t+3])
            SParsing_t = SParsing_sm_t[0].max(0, keepdim=True)[1]
            SParsing_t[SParsing_t == 1] = 100
            SParsing_t[SParsing_t == 2] = 1
            SParsing_t[SParsing_t == 3] = 2
            SParsing_t[SParsing_t == 100] = 3
            SParsing_cloth_t = torch.zeros_like(SParsing_t).cuda(opt.gpu_ids[0])
            SParsing_cloth_t[(SParsing_t == 1) | (SParsing_t == 2)] = SParsing_t[(SParsing_t == 1) | (SParsing_t == 2)]
            SParsing_cloth = torch.cat([SParsing_cloth[:,1:], SParsing_cloth_t.float().unsqueeze(0).unsqueeze(0)], dim=1)
            SParsing = torch.cat([SParsing[:,1:], SParsing_t.float().unsqueeze(0).unsqueeze(0)], dim=1)

            SPose_cloth_t = -torch.ones_like(SPose[0, t+2]).cuda(opt.gpu_ids[0])
            SPose_cloth_t[0][(SParsing_cloth_t[0] != 0)] = SPose[0, t+2][0][(SParsing_cloth_t[0] != 0)]
            SPose_cloth_t[1][(SParsing_cloth_t[0] != 0)] = SPose[0, t+2][1][(SParsing_cloth_t[0] != 0)]
            SPose_cloth_t[2][(SParsing_cloth_t[0] != 0)] = SPose[0, t+2][2][(SParsing_cloth_t[0] != 0)]
            SPose_cloth = torch.cat([SPose_cloth[:,1:], SPose_cloth_t.unsqueeze(0).unsqueeze(0)], dim=1)
    else:
        SParsing_sm_t, input_T_stage1, input_S_stage1 = Parser.inference(TParsing, SPose[:,-3:])
        SParsing_t = SParsing_sm_t[0].max(0, keepdim=True)[1]
        SParsing_t[SParsing_t == 1] = 100
        SParsing_t[SParsing_t == 2] = 1
        SParsing_t[SParsing_t == 3] = 2
        SParsing_t[SParsing_t == 100] = 3
        SParsing_cloth_t = torch.zeros_like(SParsing_t).cuda(opt.gpu_ids[0])
        SParsing_cloth_t[(SParsing_t == 1) | (SParsing_t == 2)] = SParsing_t[(SParsing_t == 1) | (SParsing_t == 2)]
        SParsing_cloth = torch.cat([SParsing_cloth[:,1:], SParsing_cloth_t.float().unsqueeze(0).unsqueeze(0)], dim=1)
        SParsing = torch.cat([SParsing[:,1:], SParsing_t.float().unsqueeze(0).unsqueeze(0)], dim=1)

        SPose_cloth_t = -torch.ones_like(SPose[0, t+2]).cuda(opt.gpu_ids[0])
        SPose_cloth_t[0][(SParsing_cloth_t[0] != 0)] = SPose[0, t+2][0][(SParsing_cloth_t[0] != 0)]
        SPose_cloth_t[1][(SParsing_cloth_t[0] != 0)] = SPose[0, t+2][1][(SParsing_cloth_t[0] != 0)]
        SPose_cloth_t[2][(SParsing_cloth_t[0] != 0)] = SPose[0, t+2][2][(SParsing_cloth_t[0] != 0)]
        SPose_cloth = torch.cat([SPose_cloth[:,1:], SPose_cloth_t.unsqueeze(0).unsqueeze(0)], dim=1)


    SFG_cloth_tps, SFG_cloth, SLO_cloth_tps, SLO_cloth, Flow_tps, Flow_dense, Flow, input_T_stage2, input_S_stage2 = ClothWarper.inference(TParsing_cloth, TFG_cloth, SParsing_cloth)

    SI, SI_raw, SFG_res, SD, SFG, input_T_stage3, input_S_stage3 = Composer.inference(TParsing_uncloth, TFG_uncloth, SPose[:,-3:], SParsing, SFG_cloth.unsqueeze(0), BG[:,-3:])

    input_tlo_cloth_ = input_T_stage2[-1-opt.label_nc_2:-1]
    input_tlo_cloth = util.tensor2lo(input_tlo_cloth_, opt.label_nc_2)
    input_tlo_uncloth_ = input_T_stage3[-3-(opt.label_nc_3-opt.label_nc_2+1):-3]
    input_tlo = util.tensor2lo(input_T_stage1, opt.label_nc_3, old_type=True)
    input_tfg_uncloth_ = input_T_stage3[-3:]
    input_tfg_uncloth_[input_tlo_uncloth_[1:].sum(dim=0, keepdim=True).expand_as(input_tfg_uncloth_)==0] = 1
    input_tfg_uncloth = util.tensor2im(input_tfg_uncloth_)
    input_tfg_cloth_ = torch.ones_like(input_T_stage2[-1:])
    input_tfg_cloth_[input_tlo_cloth_[1:].sum(dim=0, keepdim=True).expand_as(input_tfg_cloth_)!=0] = input_T_stage2[-1:][input_tlo_cloth_[1:].sum(dim=0, keepdim=True).expand_as(input_tfg_cloth_)!=0]
    input_tfg_cloth = util.tensor2im(input_tfg_cloth_)
    input_spose_ = input_S_stage3[0:3]
    input_spose_[input_spose_==-1] = 1
    input_spose = util.tensor2im(input_spose_)

    input_bg = util.tensor2im(BG[0, -1])

    output_SLO_ = input_S_stage3[-opt.label_nc_3:]
    output_SLO = util.tensor2lo(output_SLO_, opt.label_nc_3)
    output_SLO_cloth_ = input_S_stage2[-opt.label_nc_2:]
    output_SLO_cloth = util.tensor2lo(output_SLO_cloth_, opt.label_nc_2)

    output_SFG_cloth_tps_ = torch.ones_like(SFG_cloth_tps)
    output_SFG_cloth_tps_[SLO_cloth_tps.max(1, keepdim=True)[1].expand_as(output_SFG_cloth_tps_)!=0] = SFG_cloth_tps[SLO_cloth_tps.max(1, keepdim=True)[1].expand_as(output_SFG_cloth_tps_)!=0]
    output_SFG_cloth_tps = util.tensor2im(output_SFG_cloth_tps_)
    output_SFG_cloth_ = torch.ones_like(SFG_cloth)
    output_SFG_cloth_[(SLO_cloth.max(1, keepdim=True)[1].expand_as(output_SFG_cloth_)!=0) & (SParsing_cloth[0,-1].expand_as(output_SFG_cloth_)!=0)] = SFG_cloth[(SLO_cloth.max(1, keepdim=True)[1].expand_as(output_SFG_cloth_)!=0) & (SParsing_cloth[0,-1].expand_as(output_SFG_cloth_)!=0)]
    output_SFG_cloth = util.tensor2im(output_SFG_cloth_)
    output_SLO_cloth_stage2 = util.tensor2lo(SLO_cloth, opt.label_nc_2)
    output_flow_tps = fz.convert_from_flow(Flow_tps[0].permute(1,2,0).cpu().numpy())
    output_flow_dense = fz.convert_from_flow(Flow_dense[0].permute(1,2,0).cpu().numpy())
    output_flow = fz.convert_from_flow(Flow[0].permute(1,2,0).cpu().numpy())

    output_SFG_ = torch.ones_like(SFG)
    output_SFG_[output_SLO_.max(0, keepdim=True)[1].expand_as(output_SFG_)!=0] = SFG[output_SLO_.max(0, keepdim=True)[1].expand_as(output_SFG_)!=0]
    output_SFG = util.tensor2im(output_SFG_)
    output_SFG_res = util.tensor2im(SFG_res)
    SFG_res[output_SLO_[1:opt.label_nc_2].sum(dim=0, keepdim=True).expand_as(SFG_res)!=0] = 1
    output_SFG_res_uncloth = util.tensor2im(SFG_res)
    output_sd = util.tensor2im(SD)
    output_SI = util.tensor2im(SI)
    output_SI_raw = util.tensor2im(SI_raw)

    source_ref = util.tensor2im(SI_ref[0, -1])

    visual_list_input_1 = [('input_spose', input_spose),
                               ('input_tlo', input_tlo)]

    visual_list_output_1 = [('output_slo', output_SLO),
                                ('output_slo_cloth', output_SLO_cloth)]

    visual_list_input_2 = [('input_tlo_cloth', input_tlo_cloth), 
                   ('input_tfg_cloth', input_tfg_cloth)]

    visual_list_output_2 = [('output_sfg_cloth_tps', output_SFG_cloth_tps), 
                   ('output_sfg_cloth', output_SFG_cloth), 
                   ('output_SLO_cloth_stage2', output_SLO_cloth_stage2),
                   ('output_flow_tps', output_flow_tps),
                   ('output_flow_dense', output_flow_dense),
                   ('output_flow', output_flow)]

    visual_list_input_3 = [('input_tfg_uncloth', input_tfg_uncloth),
                           ('input_bg', input_bg)]

    visual_list_output_3 = [('output_sfg', output_SFG), 
                   ('output_sfg_res', output_SFG_res),
                   ('output_sfg_res_uncloth', output_SFG_res_uncloth),
                   ('output_sd', output_sd)]

    visual_list_output_4 = [('output_SI', output_SI)]
    visual_list_output_5 = [('output_SI_raw', output_SI_raw)]

    visual_list_ref = [('source_ref', source_ref)]

    visuals_input_1 = OrderedDict(visual_list_input_1) 
    visuals_output_1 = OrderedDict(visual_list_output_1) 
    visuals_input_2 = OrderedDict(visual_list_input_2) 
    visuals_output_2 = OrderedDict(visual_list_output_2) 
    visuals_input_3 = OrderedDict(visual_list_input_3) 
    visuals_output_3 = OrderedDict(visual_list_output_3) 
    visuals_output_4 = OrderedDict(visual_list_output_4) 
    visuals_output_5 = OrderedDict(visual_list_output_5) 
    visual_ref = OrderedDict(visual_list_ref)
    img_path = data['A_path']
    print('process image... %s' % img_path)
    visualizer.save_images(save_dir_input_1, visuals_input_1, img_path)
    visualizer.save_images(save_dir_output_1, visuals_output_1, img_path)
    visualizer.save_images(save_dir_input_2, visuals_input_2, img_path)
    visualizer.save_images(save_dir_output_2, visuals_output_2, img_path)
    visualizer.save_images(save_dir_input_3, visuals_input_3, img_path)
    visualizer.save_images(save_dir_output_3, visuals_output_3, img_path)
    visualizer.save_images(save_dir_output_4, visuals_output_4, img_path)
    visualizer.save_images(save_dir_output_5, visuals_output_5, img_path)
    visualizer.save_images(save_dir_ref, visual_ref, img_path)