python test_all_stages.py --name full_256p \
--dataroot SoloDance/test --dataset_mode full --model full --nThreads 16 \
--input_nc_T_1 12 --input_nc_S_1 3 --input_nc_T_2 4 --input_nc_S_2 3 --input_nc_P_2 10 --input_nc_T_3 13 --input_nc_S_3 15 \
--label_nc_1 12 --label_nc_2 3 --label_nc_3 12 --output_nc_1 12 --output_nc_2 2 --output_nc_3 3 \
--ngf 64 --n_downsample_warper 4 --grid_size 3 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 \
--no_first_img --gpu_ids 0