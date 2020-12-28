python train_stage2.py --name clothwarp_256p \
--dataroot SoloDance/train --dataset_mode cloth --model cloth --nThreads 16 \
--input_nc_T_2 4 --input_nc_S_2 3 --input_nc_P_2 10 --ngf 64 --n_downsample_warper 4 --label_nc_2 3 --grid_size 3 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 --color_aug \
--gpu_ids 0 --n_gpus_gen 1 --batchSize 1 --max_frames_per_gpu 12 --display_freq 40 --print_freq 40 --save_latest_freq 1000 \
--niter 1 --niter_decay 0 --n_scales_temporal 3 --n_frames_D 2 \
--no_first_img --n_frames_total 12 --max_t_step 4 --tf_log --tps_only