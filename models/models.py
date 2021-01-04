### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import torch
import torch.nn as nn
import numpy as np
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

def wrap_model_parser(opt, modelG, modelD, flowNet):
    if opt.n_gpus_gen == len(opt.gpu_ids):
        modelG = myModel(opt, modelG)
        modelD = myModel(opt, modelD)
        flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
        modelD = nn.DataParallel(modelD, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return modelG, modelD, flowNet

def wrap_model(opt, modelG, modelD, flowNet):
    if opt.n_gpus_gen == len(opt.gpu_ids):
        modelG = myModel(opt, modelG)
        modelD = myModel(opt, modelD)
        flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
        modelD = nn.DataParallel(modelD, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return modelG, modelD, flowNet

def wrap_model_cloth(opt, ClothWarper, ClothWarperLoss, flowNet):
    if opt.n_gpus_gen == len(opt.gpu_ids):
        ClothWarper = myModel(opt, ClothWarper)
        ClothWarperLoss = myModel(opt, ClothWarperLoss)
        flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            ClothWarper = nn.DataParallel(ClothWarper, device_ids=opt.gpu_ids[0:1])                
        else:
            gpu_split_id = opt.n_gpus_gen
            ClothWarper = nn.DataParallel(ClothWarper, device_ids=opt.gpu_ids[:gpu_split_id])
        ClothWarperLoss = nn.DataParallel(ClothWarperLoss, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return ClothWarper, ClothWarperLoss, flowNet

class myModel(nn.Module):
    def __init__(self, opt, model):        
        super(myModel, self).__init__()
        self.opt = opt
        self.module = model
        self.model = nn.DataParallel(model, device_ids=opt.gpu_ids)
        self.bs_per_gpu = int(np.ceil(float(opt.batchSize) / len(opt.gpu_ids))) # batch size for each GPU
        self.pad_bs = self.bs_per_gpu * len(opt.gpu_ids) - opt.batchSize           

    def forward(self, *inputs, **kwargs):
        inputs = self.add_dummy_to_tensor(inputs, self.pad_bs)
        outputs = self.model(*inputs, **kwargs, dummy_bs=self.pad_bs)
        if self.pad_bs == self.bs_per_gpu: # gpu 0 does 0 batch but still returns 1 batch
            return self.remove_dummy_from_tensor(outputs, 1)
        return outputs        

    def add_dummy_to_tensor(self, tensors, add_size=0):        
        if add_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
                
        if isinstance(tensors, torch.Tensor):            
            dummy = torch.zeros_like(tensors)[:add_size]
            tensors = torch.cat([dummy, tensors])
        return tensors

    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if remove_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
        
        if isinstance(tensors, torch.Tensor):
            tensors = tensors[remove_size:]
        return tensors

def create_model(opt):    
    print(opt.model)            
    if opt.model == 'composer':
        from .modelG_stage3 import Vid2VidModelG
        modelG = Vid2VidModelG()    
        if opt.isTrain:
            from .modelD_stage3 import Vid2VidModelD
            modelD = Vid2VidModelD()    
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.isTrain:
        from .flownet import FlowNet
        flowNet = FlowNet()
    
    modelG.initialize(opt)
    if opt.isTrain:
        modelD.initialize(opt)
        flowNet.initialize(opt)        
        if not opt.fp16:
            modelG, modelD, flownet = wrap_model(opt, modelG, modelD, flowNet)
        return [modelG, modelD, flowNet]
    else:
        return modelG

def create_model_full(opt):    
    print(opt.model)            
    if opt.model == 'full':
        from .modelG_stage1 import Vid2VidModelG as Parser
        Parser = Parser()
        from .model_stage2 import ClothWarper
        ClothWarper = ClothWarper()
        from .modelG_stage3 import Vid2VidModelG as Composer
        Composer = Composer()    
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    
    import copy
    opt_1 = copy.deepcopy(opt)
    opt_1.name = 'parser_256p'
    opt_2 = copy.deepcopy(opt)
    opt_2.name = 'clothwarp_256p'
    opt_3 = copy.deepcopy(opt)
    opt_3.name = 'composer_256p'
    Parser.initialize(opt_1)
    ClothWarper.initialize(opt_2)
    Composer.initialize(opt_3)

    return Parser, ClothWarper, Composer

def create_model_parser(opt):    
    print(opt.model)            
    if opt.model == 'parser':
        from .modelG_stage1 import Vid2VidModelG
        modelG = Vid2VidModelG()    
        if opt.isTrain:
            from .modelD_stage1 import Vid2VidModelD
            modelD = Vid2VidModelD()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    
    if opt.isTrain:
        from .flownet import FlowNet
        flowNet = FlowNet()

    modelG.initialize(opt)
    if opt.isTrain:
        modelD.initialize(opt)
        flowNet.initialize(opt)
        if not opt.fp16:
            modelG, modelD, flowNet = wrap_model_parser(opt, modelG, modelD, flowNet)
        return [modelG, modelD, flowNet]
    else:
        return modelG

def create_model_cloth(opt):    
    print(opt.model)              
    if opt.model == 'cloth':
        from .model_stage2 import ClothWarper
        ClothWarper = ClothWarper()
        if opt.isTrain:
            from .loss_stage2 import ClothWarperLoss
            ClothWarperLoss = ClothWarperLoss()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.isTrain:
        from .flownet import FlowNet
        flowNet = FlowNet()
    
    ClothWarper.initialize(opt)
    if opt.isTrain:
        ClothWarperLoss.initialize(opt)
        flowNet.initialize(opt)        
        if not opt.fp16:
            ClothWarper, ClothWarperLoss, flownet = wrap_model_cloth(opt, ClothWarper, ClothWarperLoss, flowNet)
        return [ClothWarper, ClothWarperLoss, flowNet]
    else:
        return ClothWarper

def create_optimizer_cloth(opt, models):
    ClothWarper, ClothWarperLoss, flowNet = models  
    if opt.fp16:              
        from apex import amp
        ClothWarper, optimizer = amp.initialize(ClothWarper, ClothWarper.optimizer, opt_level='O1')      
        ClothWarper, flownet = wrap_model_cloth(opt, ClothWarper, flowNet)
    else:        
        optimizer = ClothWarper.module.optimizer
    return ClothWarper, ClothWarperLoss, flowNet, optimizer

def create_optimizer(opt, models):
    modelG, modelD, flowNet = models
    optimizer_D_T = []    
    if opt.fp16:              
        from apex import amp
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD, 'optimizer_D_T'+str(s)))
        modelG, optimizer_G = amp.initialize(modelG, modelG.optimizer_G, opt_level='O1')
        modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D] + optimizer_D_T, opt_level='O1')
        optimizer_D, optimizer_D_T = optimizers_D[0], optimizers_D[1:]        
        modelG, modelD, flowNet = wrap_model(opt, modelG, modelD, flowNet)
    else:        
        optimizer_G = modelG.module.optimizer_G
        optimizer_D = modelD.module.optimizer_D        
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD.module, 'optimizer_D_T'+str(s)))
    return modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T

def create_optimizer_parser(opt, models):
    modelG, modelD, flowNet = models
    optimizer_D_T = []    
    if opt.fp16:              
        from apex import amp
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD, 'optimizer_D_T'+str(s)))
        modelG, optimizer_G = amp.initialize(modelG, modelG.optimizer_G, opt_level='O1')
        modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D] + optimizer_D_T, opt_level='O1')
        optimizer_D, optimizer_D_T = optimizers_D[0], optimizers_D[1:]        
        modelG, modelD, flowNet = wrap_model_parser(opt, modelG, modelD, flowNet)
    else:        
        optimizer_G = modelG.module.optimizer_G
        optimizer_D = modelD.module.optimizer_D        
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD.module, 'optimizer_D_T'+str(s)))
    return modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T

def init_params_composer(opt, modelG, modelD, data_loader):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    ### if continue training, recover previous states
    if opt.continue_train:        
        if os.path.exists(iter_path):
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > opt.niter:
            modelG.module.update_learning_rate(start_epoch-1, 'G')
            modelD.module.update_learning_rate(start_epoch-1, 'D')
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (start_epoch > opt.niter_fix_global):
            modelG.module.update_fixed_params()

    n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1   # number of gpus used for generator for each batch
    tG, tD = opt.n_frames_G, opt.n_frames_D
    tDB = tD * 3    
    s_scales = opt.n_scales_spatial
    t_scales = opt.n_scales_temporal
    input_nc_1 = opt.input_nc_T_3
    input_nc_2 = opt.input_nc_S_3

    print_freq = lcm(opt.print_freq, opt.batchSize)
    total_steps = (start_epoch-1) * len(data_loader) + epoch_iter
    total_steps = total_steps // print_freq * print_freq  

    return n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc_1, input_nc_2, start_epoch, epoch_iter, print_freq, total_steps, iter_path

def init_params(opt, ClothWarper, data_loader):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    ### if continue training, recover previous states
    if opt.continue_train:        
        if os.path.exists(iter_path):
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > opt.niter:
            ClothWarper.module.update_learning_rate_cloth(start_epoch-1)

    n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1   # number of gpus used for generator for each batch
    tG = opt.n_frames_G
    tD = opt.n_frames_D

    t_scales = opt.n_scales_temporal

    input_nc_1 = opt.input_nc_T_2
    input_nc_2 = opt.input_nc_S_2
    input_nc_3 = opt.input_nc_P_2

    print_freq = lcm(opt.print_freq, opt.batchSize)
    total_steps = (start_epoch-1) * len(data_loader) + epoch_iter
    total_steps = total_steps // print_freq * print_freq  

    return n_gpus, tG, input_nc_1, input_nc_2, input_nc_3, start_epoch, epoch_iter, print_freq, total_steps, iter_path, tD, t_scales

def save_models_cloth(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, ClothWarper, end_of_epoch=False):
    if not end_of_epoch:
        if total_steps % opt.save_latest_freq == 0:
            visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            ClothWarper.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    else:
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            ClothWarper.module.save('latest')
            ClothWarper.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

def save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=False):
    if not end_of_epoch:
        if total_steps % opt.save_latest_freq == 0:
            visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    else:
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            modelG.module.save('latest')
            modelD.module.save('latest')
            modelG.module.save(epoch)
            modelD.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

def update_models_cloth(opt, epoch, ClothWarper, data_loader):
    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ClothWarper.module.update_learning_rate_cloth(epoch)

def update_models(opt, epoch, modelG, modelD, data_loader):
    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        modelG.module.update_learning_rate(epoch, 'G')
        modelD.module.update_learning_rate(epoch, 'D')