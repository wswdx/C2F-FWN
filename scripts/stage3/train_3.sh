python train_stage3.py --name composer_256p \
--dataroot SoloDance/train --dataset_mode composer --model composer --nThreads 16 \
--input_nc_T_3 13 --input_nc_S_3 15 --ngf 64 --ndf 32 --add_face_disc --label_nc_3 12 --label_nc_2 3 --output_nc_3 3 --num_D 2 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 --color_aug \
--gpu_ids 0 --n_gpus_gen 1 --batchSize 1 --max_frames_per_gpu 4 --display_freq 40 --print_freq 40 --save_latest_freq 1000 \
--niter 5 --niter_decay 5 --n_scales_temporal 3 \
--no_first_img --n_frames_total 12 --max_t_step 4 --tf_log