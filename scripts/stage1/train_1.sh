python train_stage1.py --name parser_256p \
--dataroot SoloDance/train --dataset_mode parser --model parser --nThreads 16 \
--input_nc_T_1 12 --input_nc_S_1 3 --ngf 64 --ndf 32 --label_nc_1 12 --output_nc_1 12 --num_D 2 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 \
--gpu_ids 0 --n_gpus_gen 1 --batchSize 1 --max_frames_per_gpu 6 --display_freq 40 --print_freq 40 --save_latest_freq 1000 \
--niter 5 --niter_decay 5 --n_scales_temporal 3 --lambda_P 10 \
--no_first_img --n_frames_total 12 --max_t_step 4 --tf_log