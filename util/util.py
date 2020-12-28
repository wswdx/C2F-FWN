from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from PIL import Image
import cv2
from collections import OrderedDict
import flowiz as fz

def save_all_tensors_cloth(opt, real_input_1, real_input_2, fg_tps, fg_dense, lo_tps, lo_dense, fg_dense_first, real_SFG, real_SFG_full, flow_tps, flow_dense, flow_total):
    #print(real_input_1.size(), real_input_2.size(), lo_dense.size())
    #turn to white background:
    #layout:
    input_tlo_ = real_input_1[0, -1, -1-opt.label_nc_2:-1]
    input_slo_ = real_input_2[0, -1, -opt.label_nc_2:]
    output_slo_tps_ = lo_tps[0, -1]
    output_slo_dense_ = lo_dense[0, -1]
    #foreground:
    input_tfg_ = real_input_1[0, -1, -1:]
    input_tfg_[input_tlo_[1:].sum(dim=0, keepdim=True)==0] = 1
    real_sfg_ = real_SFG[0, -1]
    real_sfg_[input_slo_[1:].sum(dim=0, keepdim=True).expand_as(real_sfg_)==0] = 1
    real_sfg_full_ = real_SFG_full[0, -1]

    output_sfg_tps_ = fg_tps[0, -1]
    output_sfg_tps_[output_slo_tps_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_tps_)==0] = 1
    output_sfg_dense_ = fg_dense[0, -1]
    output_sfg_dense_[output_slo_dense_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_dense_)==0] = 1
    output_sfg_full_ = torch.ones_like(real_sfg_full_)
    output_sfg_full_[input_slo_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_full_)==0] = real_sfg_full_[input_slo_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_full_)==0]
    output_sfg_full_[(input_slo_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_full_)==1) & (output_slo_dense_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_full_)!=0)] = output_sfg_dense_[(input_slo_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_full_)==1) & (output_slo_dense_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_full_)!=0)]

    input_slo = tensor2lo(input_slo_, opt.label_nc_2)
    real_sfg = tensor2im(real_sfg_)
    real_sfg_full = tensor2im(real_sfg_full_)
    input_tlo = tensor2lo(input_tlo_, opt.label_nc_2)
    input_tfg = tensor2im(input_tfg_)
    output_sfg_tps = tensor2im(output_sfg_tps_)
    output_sfg_dense = tensor2im(output_sfg_dense_)
    output_slo_tps = tensor2lo(output_slo_tps_, opt.label_nc_2)
    output_slo_dense = tensor2lo(output_slo_dense_, opt.label_nc_2)
    output_sfg_first = tensor2im(fg_dense_first)
    output_sfg_full = tensor2im(output_sfg_full_)
    output_flow_tps = fz.convert_from_flow(flow_tps[0, -1].permute(1,2,0).data.cpu().numpy())
    output_flow_dense = fz.convert_from_flow(flow_dense[0, -1].permute(1,2,0).data.cpu().numpy())
    output_flow_total = fz.convert_from_flow(flow_total[0, -1].permute(1,2,0).data.cpu().numpy())
    #output_flow_tps = tensor2flow(flow_tps[0, -1])
    #output_flow_dense = tensor2flow(flow_dense[0, -1])
    #output_flow_total = tensor2flow(flow_total[0, -1])

    visual_list = [('input_slo', input_slo),
                   ('real_sfg', real_sfg),
                   ('real_sfg_full', real_sfg_full),
                   ('input_tlo', input_tlo),
                   ('input_tfg', input_tfg),
                   ('output_sfg_tps', output_sfg_tps),
                   ('output_sfg_dense', output_sfg_dense),
                   ('output_slo_tps', output_slo_tps),
                   ('output_slo_dense', output_slo_dense),
                   ('output_sfg_first', output_sfg_first),
                   ('output_sfg_full', output_sfg_full),
                   ('output_flow_tps', output_flow_tps),
                   ('output_flow_dense', output_flow_dense),
                   ('output_flow_total', output_flow_total)]
    visuals = OrderedDict(visual_list)
    return visuals

def save_all_tensors_sampled(opt, flows_sampled_0, flows_sampled_1):
    #print(real_input_1.size(), real_input_2.size(), lo_dense.size())
    #turn to white background:
    #pose:
    flow_warp_0, flow_prev_0, flow_0 = flows_sampled_0
    flow_warp_1, flow_prev_1, flow_1 = flows_sampled_1
    flow_warp_0_vis = fz.convert_from_flow(flow_warp_0[0].permute(1,2,0).data.cpu().numpy())
    flow_prev_0_vis = fz.convert_from_flow(flow_prev_0[0].permute(1,2,0).data.cpu().numpy())
    flow_0_vis = fz.convert_from_flow(flow_0[0].permute(1,2,0).data.cpu().numpy())
    flow_warp_1_vis = fz.convert_from_flow(flow_warp_1[0].permute(1,2,0).data.cpu().numpy())
    flow_prev_1_vis = fz.convert_from_flow(flow_prev_1[0].permute(1,2,0).data.cpu().numpy())
    flow_1_vis = fz.convert_from_flow(flow_1[0].permute(1,2,0).data.cpu().numpy())

    visual_list = [('0_flow_warp_vis', flow_warp_0_vis),
                   ('0_flow_prev_vis', flow_prev_0_vis),
                   ('0_flow_vis', flow_0_vis),
                   ('1_flow_warp_vis', flow_warp_1_vis),
                   ('1_flow_prev_vis', flow_prev_1_vis),
                   ('1_flow_vis', flow_1_vis)]
    visuals = OrderedDict(visual_list)
    return visuals

def save_all_tensors_composer(opt, real_input_T, real_input_S, real_input_SFG, real_input_BG, fake_SI, fake_SI_raw, fake_SI_first, fake_SFG_full, fake_SFG_res, fake_sd, real_SI, real_SFG_full, flow_ref, conf_ref, flow, weight, modelD):
    #pose:
    input_spose_ = real_input_S[0, -1, 0:3]
    input_spose_[input_spose_==-1] = 1
    #layout:
    input_tlo_ = real_input_T[0, -1, -3-(opt.label_nc_3-opt.label_nc_2+1):-3]
    input_slo_ = real_input_S[0, -1, -opt.label_nc_3:]
    #foreground:
    input_tfg_ = real_input_T[0, -1, -3:]
    #input_tfg_[input_tlo_[1:].sum(dim=0, keepdim=True).expand_as(input_tfg_)==0] = 1
    input_sfg_ = real_input_SFG[0, -1]
    #input_sfg_[input_slo_[1:opt.label_nc_2].sum(dim=0, keepdim=True).expand_as(input_sfg_)==0] = 1
    output_sfg_full_ = fake_SFG_full[0, -1]
    output_sfg_full_[input_slo_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_full_)==0] = 1
    output_sfg_res_raw_ = fake_SFG_res[0, -1]
    #output_sfg_res_raw_[input_slo_[1:].sum(dim=0, keepdim=True).expand_as(output_sfg_res_raw_)==0] = 1
    output_sfg_res_raw = tensor2im(output_sfg_res_raw_)
    output_sfg_res_ = output_sfg_res_raw_
    output_sfg_res_[input_slo_[opt.label_nc_2:].sum(dim=0, keepdim=True).expand_as(output_sfg_res_)==0] = 1
    output_sfg_res = tensor2im(output_sfg_res_)
    real_sfg_full_ = real_SFG_full[0, -1]
    real_sfg_full_[input_slo_[1:].sum(dim=0, keepdim=True).expand_as(real_sfg_full_)==0] = 1
    #full image:
    output_si_raw_ = fake_SI_raw[0, -1]
    output_si_ = fake_SI[0, -1]
    real_si_ = real_SI[0, -1]

    input_tlo = tensor2lo(input_tlo_, opt.label_nc_3-opt.label_nc_2+1)
    input_tfg = tensor2im(input_tfg_)
    input_spose = tensor2im(input_spose_)
    input_slo = tensor2lo(input_slo_, opt.label_nc_3)
    input_sfg = tensor2im(input_sfg_)
    input_bg = tensor2im(real_input_BG[0, -1])
    output_sfg_full = tensor2im(output_sfg_full_)
    output_si_raw = tensor2im(output_si_raw_)
    output_si = tensor2im(output_si_)
    output_si_first = tensor2im(fake_SI_first)
    output_sd = tensor2im(fake_sd[0, -1], normalize=False)
    output_flow = fz.convert_from_flow(flow[0, -1].permute(1,2,0).data.cpu().numpy())
    #output_flow = tensor2flow(flow[0, -1])
    output_flow_weight = tensor2im(weight[0, -1], normalize=False)
    real_sfg_full = tensor2im(real_sfg_full_)
    real_si = tensor2im(real_si_)
    real_flow = fz.convert_from_flow(flow_ref[0, -1].permute(1,2,0).cpu().numpy())
    #real_flow = tensor2flow(flow_ref[0, -1])

    if opt.add_face_disc:
        ys_T, ye_T, xs_T, xe_T = modelD.module.get_face_region(real_input_T[0, -1:, -3-(opt.label_nc_3-opt.label_nc_2+1)+2:-3-(opt.label_nc_3-opt.label_nc_2+1)+3])
        ys_S, ye_S, xs_S, xe_S = modelD.module.get_face_region(real_input_S[0, -1:, -opt.label_nc_3+opt.label_nc_2+1:-opt.label_nc_3+opt.label_nc_2+2])
        if ys_S is not None and ys_T is not None:
            input_tfg[ys_T, xs_T:xe_T, :] = input_tfg[ye_T, xs_T:xe_T, :] = input_tfg[ys_T:ye_T, xs_T, :] = input_tfg[ys_T:ye_T, xe_T, :] = 255 
            output_sfg_full[ys_S, xs_S:xe_S, :] = output_sfg_full[ye_S, xs_S:xe_S, :] = output_sfg_full[ys_S:ye_S, xs_S, :] = output_sfg_full[ys_S:ye_S, xe_S, :] = 0 

    visual_list = [('input_tlo', input_tlo),
                   ('input_tfg', input_tfg),
                   ('input_spose', input_spose),
                   ('input_slo', input_slo),
                   ('input_sfg', input_sfg),
                   ('input_bg', input_bg),
                   ('output_sfg_full', output_sfg_full),
                   ('output_sfg_res_raw', output_sfg_res_raw),
                   ('output_sfg_res', output_sfg_res),
                   ('output_si_raw', output_si_raw),
                   ('output_si', output_si),
                   ('output_si_first', output_si_first),
                   ('output_sd', output_sd),
                   ('output_flow', output_flow),
                   ('output_flow_weight', output_flow_weight),
                   ('real_sfg_full', real_sfg_full),
                   ('real_si', real_si),
                   ('real_flow', real_flow)]

    visuals = OrderedDict(visual_list)
    return visuals

def save_all_tensors_parser(opt, real_input_T, real_input_S, fake_slo, fake_slo_raw, fake_slo_first, real_slo, flow_ref, conf_ref, flow, weight):
    #pose:
    input_spose_ = real_input_S[0, -1, 0:3]
    input_spose_[input_spose_==-1] = 1
    #layout:
    input_tlo_ = real_input_T[0, -1, -opt.label_nc_1:]
    output_slo_ = fake_slo[0, -1]
    output_slo_raw_ = fake_slo_raw[0, -1]
    output_slo_first_ = fake_slo_first
    real_slo_ = real_slo[0, -1]

    input_tlo = tensor2lo(input_tlo_, opt.label_nc_1, old_type=True)
    input_spose = tensor2im(input_spose_)
    output_slo = tensor2lo(output_slo_, opt.label_nc_1, old_type=True)
    output_slo_raw = tensor2lo(output_slo_raw_, opt.label_nc_1, old_type=True)
    output_slo_first = tensor2lo(output_slo_first_, opt.label_nc_1, old_type=True)
    real_slo = tensor2lo(real_slo_, opt.label_nc_1, old_type=True)

    output_flow = fz.convert_from_flow(flow[0, -1].permute(1,2,0).data.cpu().numpy())
    output_flow_weight = tensor2im(weight[0, -1], normalize=False)
    real_flow = fz.convert_from_flow(flow_ref[0, -1].permute(1,2,0).cpu().numpy())

    visual_list = [('input_tlo', input_tlo),
                   ('input_spose', input_spose),
                   ('output_slo', output_slo),
                   ('output_slo_raw', output_slo_raw),
                   ('output_flow', output_flow),
                   ('output_flow_weight', output_flow_weight),
                   ('output_slo_first', output_slo_first),
                   ('real_slo', real_slo),
                   ('real_flow', real_flow)]

    visuals = OrderedDict(visual_list)
    return visuals

def save_all_tensors(opt, real_A, fake_B, fake_B_first, fake_B_raw, real_B, flow_ref, conf_ref, flow, weight, modelD):
    if opt.label_nc != 0:
        input_image = tensor2label(real_A, opt.label_nc)
    elif opt.dataset_mode == 'pose':
        input_image = tensor2im(real_A)
        if real_A.size()[2] == 6:
            input_image2 = tensor2im(real_A[0, -1, 3:])
            input_image[input_image2 != 0] = input_image2[input_image2 != 0]
    else:
        c = 3 if opt.input_nc >= 3 else 1
        input_image = tensor2im(real_A[0, -1, :c], normalize=False)
    if opt.use_instance:
        edges = tensor2im(real_A[0, -1, -1:], normalize=False)
        input_image += edges[:,:,np.newaxis]
    
    if opt.add_face_disc:
        ys, ye, xs, xe = modelD.module.get_face_region(real_A[0, -1:])
        if ys is not None:
            input_image[ys, xs:xe, :] = input_image[ye, xs:xe, :] = input_image[ys:ye, xs, :] = input_image[ys:ye, xe, :] = 255 

    visual_list = [('input_image', input_image),
                   ('fake_image', tensor2im(fake_B)),
                   ('fake_first_image', tensor2im(fake_B_first)),
                   ('fake_raw_image', tensor2im(fake_B_raw)),
                   ('real_image', tensor2im(real_B)),                                                          
                   ('flow_ref', tensor2flow(flow_ref)),
                   ('conf_ref', tensor2im(conf_ref, normalize=False))]
    if flow is not None:
        visual_list += [('flow', tensor2flow(flow)),
                        ('weight', tensor2im(weight, normalize=False))]
    visuals = OrderedDict(visual_list)
    return visuals

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor[0, -1]
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor[:3]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tensor2loraw(output, imtype=np.uint8):
    if isinstance(output, list):
        output_numpy = []
        for i in range(len(output)):
            output_numpy.append(tensor2loraw(output[i], np.uint8))
        return output_numpy
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float()
    if output.size()[0] > 1:
        output = output.max(0, keepdim=True)[1]

    output_numpy = output.numpy()
    output_numpy = np.transpose(output_numpy, (1, 2, 0))
    if output_numpy.shape[2] == 1:        
        output_numpy = output_numpy[:,:,0]
    return output_numpy.astype(imtype)

def tensor2lo(output, n_label, imtype=np.uint8, old_type=False):
    if isinstance(output, list):
        output_numpy = []
        for i in range(len(output)):
            output_numpy.append(tensor2lo(output[i], n_label, np.uint8))
        return output_numpy
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float()
    if output.size()[0] > 1:
        output = output.max(0, keepdim=True)[1]
    output = Colorize(n=n_label, old_type=old_type)(output)
    output_numpy = output.numpy()
    output_numpy = np.transpose(output_numpy, (1, 2, 0))
    if output_numpy.shape[2] == 1:        
        output_numpy = output_numpy[:,:,0]
    return output_numpy.astype(imtype)

def tensor2label(output, n_label, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float()    
    if output.size()[0] > 1:
        output = output.max(0, keepdim=True)[1]
    #print(output.size())
    output = Colorize(n_label)(output)
    output = np.transpose(output.numpy(), (1, 2, 0))
    #img = Image.fromarray(output, "RGB")
    return output.astype(imtype)

def tensor2flow(output, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float().numpy()
    output = np.transpose(output, (1, 2, 0))
    #mag = np.max(np.sqrt(output[:,:,0]**2 + output[:,:,1]**2)) 
    #print(mag)
    hsv = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(output[..., 0], output[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def add_dummy_to_tensor(tensors, add_size=0):
    if add_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        dummy = torch.zeros_like(tensors)[:add_size]
        tensors = torch.cat([dummy, tensors])
    return tensors

def remove_dummy_from_tensor(tensors, remove_size=0):
    if remove_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N, old_type=False):
    if N == 3:
        cmap = np.array([(255,255,255), (255,85,0), (0,85,85)], dtype=np.uint8)
    elif N == 4:
        cmap = np.array([(255,255,255), (255,0,0), (255,85,0), (0,85,85)], dtype=np.uint8)
    elif N == 5:
        cmap = np.array([(255,255,255), (255,0,0), (255,85,0), (0,85,85), (0,128,0)], dtype=np.uint8)
    elif N == 6:
        cmap = np.array([(255,255,255), (255,0,0), (255,85,0), (0,85,85), (255,255,0), (255,170,0)], dtype=np.uint8)
    elif N == 7:
        cmap = np.array([(255,255,255), (255,0,0), (255,85,0), (0,85,85), (0,128,0), (255,255,0), (255,170,0)], dtype=np.uint8)
    elif N == 10:
        cmap = np.array([(255,255,255), (255,0,0), (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (85,51,0), (255,255,0), (255,170,0)], dtype=np.uint8)
    #elif N == 9:
    #    cmap = np.array([(255,255,255), (255,0,0), (255,85,0), (0,85,85), (0,0,255), (51,170,221), (85,255,170), (85,51,0), (255,255,0)], dtype=np.uint8)
    elif N == 12 and not old_type:
        cmap = np.array([(255,255,255), (255,85,0), (0,85,85), (255,0,0), (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (85,51,0), (255,255,0), (255,170,0)], dtype=np.uint8)
    elif N == 12 and old_type:
        cmap = np.array([(255,255,255), (255,0,0), (255,85,0), (0,85,85), (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (85,51,0), (255,255,0), (255,170,0)], dtype=np.uint8)
    elif N == 13:
        cmap = np.array([(255,255,255), (255,0,0), (255,85,0), (0,85,85), (0,128,0), (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0), (85,51,0)], dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0            
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0], cmap[i, 1], cmap[i, 2] = r, g, b             
    return cmap

def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7-j))*((i & (1 << (3*j))) >> (3*j))
            g = g + (1 << (7-j))*((i & (1 << (3*j+1))) >> (3*j+1))
            b = b + (1 << (7-j))*((i & (1 << (3*j+2))) >> (3*j+2))

        cmap[i, :] = np.array([r, g, b])

    return cmap

class Colorize(object):
    def __init__(self, n=35, old_type=False):
        self.cmap = labelcolormap(n, old_type)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image