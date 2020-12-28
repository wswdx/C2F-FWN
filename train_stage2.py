### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import torch
from subprocess import call

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model, create_optimizer, init_params, save_models_cloth, update_models_cloth, create_model_cloth, create_optimizer_cloth
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
    models = create_model_cloth(opt)
    ClothWarper, ClothWarperLoss, flowNet, optimizer = create_optimizer_cloth(opt, models)

    ### set parameters    
    n_gpus, tG, input_nc_1, input_nc_2, input_nc_3, start_epoch, epoch_iter, print_freq, total_steps, iter_path, tD, t_scales = init_params(opt, ClothWarper, data_loader)
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
            n_frames_total, n_frames_load, t_len = data_loader.dataset.init_data_params_cloth(data, n_gpus, tG)
            flow_total_prev_last, frames_all = data_loader.dataset.init_data_cloth(t_scales)

            for i in range(0, n_frames_total, n_frames_load):
                is_first_frame = flow_total_prev_last is None
                input_TParsing, input_TFG, input_SParsing, input_SFG, input_SFG_full = data_loader.dataset.prepare_data_cloth(data, i)

                ###################################### Forward Pass ##########################
                ####### C2F-FWN                  
                fg_tps, fg_dense, lo_tps, lo_dense, flow_tps, flow_dense, flow_totalp, real_input_1, real_input_2, real_SFG, real_SFG_fullp, flow_total_last = ClothWarper(input_TParsing, input_TFG, input_SParsing, input_SFG, input_SFG_full, flow_total_prev_last)
                real_SLO = real_input_2[:, :, -opt.label_nc_2:]
                ####### compute losses
                ### individual frame losses and FTC loss with l=1
                real_SFG_full_prev, real_SFG_full = real_SFG_fullp[:, :-1], real_SFG_fullp[:, 1:]   # the collection of previous and current real frames
                flow_optical_ref, conf_optical_ref = flowNet(real_SFG_full, real_SFG_full_prev)       # reference flows and confidences                
                
                flow_total_prev, flow_total = flow_totalp[:, :-1], flow_totalp[:, 1:]
                if is_first_frame:
                    flow_total_prev = flow_total_prev[:, 1:]

                flow_total_prev_last = flow_total_last
                
                losses, flows_sampled_0 = ClothWarperLoss(0, reshape([real_SFG, real_SLO, fg_tps, fg_dense, lo_tps, lo_dense, flow_tps, flow_dense, flow_total, flow_total_prev, flow_optical_ref, conf_optical_ref]), is_first_frame)
                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(ClothWarperLoss.module.loss_names, losses))          

                ### FTC losses with l=3,9
                # get skipped frames for each temporal scale
                frames_all, frames_skipped = ClothWarperLoss.module.get_all_skipped_frames(frames_all, \
                        real_SFG_full, flow_total, flow_optical_ref, conf_optical_ref, real_SLO, t_scales, tD, n_frames_load, i, flowNet)                                

                # compute losses for l=3,9
                loss_dict_T = []
                for s in range(1, t_scales):                
                    if frames_skipped[0][s] is not None and not opt.tps_only:                        
                        losses, flows_sampled_1 = ClothWarperLoss(s+1, [frame_skipped[s] for frame_skipped in frames_skipped], False)
                        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                        loss_dict_T.append(dict(zip(ClothWarperLoss.module.loss_names_T, losses)))                  

                # collect losses
                loss, _ = ClothWarperLoss.module.get_losses(loss_dict, loss_dict_T, t_scales-1)

                ###################################### Backward Pass #################################                 
                # update generator weights     
                loss_backward(opt, loss, optimizer)                

                if i == 0: fg_dense_first = fg_dense[0, 0]   # the first generated image in this sequence


            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                for s in range(len(loss_dict_T)):
                    errors.update({k+str(s): v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T[s].items()})

                loss_names_vis = ClothWarperLoss.module.loss_names.copy()
                {loss_names_vis.append(ClothWarperLoss.module.loss_names_T[0]+str(idx)) for idx in range(len(loss_dict_T))}
                visualizer.print_current_errors_new(epoch, epoch_iter, errors, loss_names_vis, t)
                visualizer.plot_current_errors(errors, total_steps)
            ### display output images
            if save_fake:                
                visuals = util.save_all_tensors_cloth(opt, real_input_1, real_input_2, fg_tps, fg_dense, lo_tps, lo_dense, fg_dense_first, real_SFG, real_SFG_full, flow_tps, flow_dense, flow_total)            
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            save_models_cloth(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, ClothWarper)            
            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch and update model params
        save_models_cloth(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, ClothWarper, end_of_epoch=True)
        update_models_cloth(opt, epoch, ClothWarper, data_loader) 

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