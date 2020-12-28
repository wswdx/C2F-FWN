### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import torch
from subprocess import call

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model, create_optimizer, init_params_composer, save_models, update_models
import util.util as util
from util.visualizer import Visualizer

def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1

    ### initialize dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)    
    print('#training frames = %d' % dataset_size)

    ### initialize models
    models = create_model(opt)
    modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T = create_optimizer(opt, models)

    ### set parameters    
    n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc_1, input_nc_2, \
        start_epoch, epoch_iter, print_freq, total_steps, iter_path = init_params_composer(opt, modelG, modelD, data_loader)
    visualizer = Visualizer(opt)    

    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()    
        for idx, data in enumerate(dataset, start=epoch_iter):        
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0
            n_frames_total, n_frames_load, t_len = data_loader.dataset.init_data_params(data, n_gpus, tG)
            fake_SI_prev_last, frames_all = data_loader.dataset.init_data(t_scales)

            for i in range(0, n_frames_total, n_frames_load):
                input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_BG, input_SFG_full, input_SI = data_loader.dataset.prepare_data_composer(data, i)

                ###################################### Forward Pass ##########################
                ####### generator                  
                fake_SI, fake_SI_raw, fake_sd, fake_SFG_full, fake_SFG_res, flow, weight, real_input_T, real_input_S, real_input_SFG, real_input_BG, real_SIp, real_SFG_fullp, fake_SI_last \
                    = modelG(input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_BG, input_SFG_full, input_SI, fake_SI_prev_last)

                ####### discriminator            
                ### individual frame discriminator          
                real_SI_prev, real_SI = real_SIp[:, :-1], real_SIp[:, 1:]   # the collection of previous and current real frames
                real_SFG_full_prev, real_SFG_full = real_SFG_fullp[:, :-1], real_SFG_fullp[:, 1:]
                #flow_ref, conf_ref = flowNet(real_SI, real_SI_prev)       # reference flows and confidences                
                flow_ref, conf_ref = flowNet(real_SFG_full, real_SFG_full_prev)       # reference flows and confidences                
                fake_SI_prev = modelG.module.compute_fake_B_prev(real_SI_prev, fake_SI_prev_last, fake_SI)
                fake_SI_prev_last = fake_SI_last

                real_input_BG_flag = data['BG_flag']
                losses = modelD(0, [real_SI, real_SFG_full, fake_SI, fake_SI_raw, fake_SFG_full, fake_SFG_res, real_input_T, real_input_S, real_input_SFG, real_input_BG, real_input_BG_flag, real_SI_prev, fake_SI_prev, real_SFG_full_prev, flow, weight, flow_ref, conf_ref])
                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(modelD.module.loss_names, losses))

                ### temporal discriminator                
                # get skipped frames for each temporal scale
                frames_all, frames_skipped = modelD.module.get_all_skipped_frames(frames_all, \
                        real_SI, fake_SI, flow_ref, conf_ref, t_scales, tD, n_frames_load, i, flowNet)                                

                # run discriminator for each temporal scale
                loss_dict_T = []
                for s in range(t_scales):                
                    if frames_skipped[0][s] is not None:                        
                        losses = modelD(s+1, [frame_skipped[s] for frame_skipped in frames_skipped])
                        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                        loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))

                # collect losses
                loss_G, loss_D, loss_D_T, t_scales_act = modelD.module.get_losses(loss_dict, loss_dict_T, t_scales)

                ###################################### Backward Pass #################################                 
                # update generator weights     
                loss_backward(opt, loss_G, optimizer_G)                

                # update individual discriminator weights                
                loss_backward(opt, loss_D, optimizer_D)

                # update temporal discriminator weights
                for s in range(t_scales_act):                    
                    loss_backward(opt, loss_D_T[s], optimizer_D_T[s])

                if i == 0: fake_SI_first = fake_SI[0, 0]   # the first generated image in this sequence

            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                for s in range(len(loss_dict_T)):
                    errors.update({k+str(s): v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T[s].items()})            
                visualizer.print_current_errors_new(epoch, epoch_iter, errors, modelD.module.loss_names, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:                
                visuals = util.save_all_tensors_composer(opt, real_input_T, real_input_S, real_input_SFG, real_input_BG, fake_SI, fake_SI_raw, fake_SI_first, fake_SFG_full, fake_SFG_res, fake_sd, real_SI, real_SFG_full, flow_ref, conf_ref, flow, weight, modelD)                
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD)            
            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch and update model params
        save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=True)
        update_models(opt, epoch, modelG, modelD, data_loader) 

def loss_backward(opt, loss, optimizer):
    optimizer.zero_grad()                
    if opt.fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss: 
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

def reshape(tensors):
    if tensors is None: return None
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]    
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

if __name__ == "__main__":
   train()