#! /bin/bash

cd ..

options=" \
      --deepspeed \
      --deepspeed_config cfg/deepspeed_config.json \
      --logging_enabled False \
      --all_view_combinations False \
      --n_inputs 6 \
      --pre_trained_i3d_img /home/ola/Projects/SUEF/pretrained_models/i3d_bert_rgb_imagenet.pth \
      --pre_trained_i3d_flow /home/ola/Projects/SUEF/pretrained_models/i3d_bert_flow_imagenet.pth \
      --project_name eiphodos/SUEF \
      --temp_folder_img /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/temp/img \
      --temp_folder_flow /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/temp/flow \
      --train_targets /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/train_data_mod5.csv \
      --val_targets /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/targets/val_data_mod5.csv \
      --allowed_views 0 2 4 \
      --data_folder_img /media/ola/7540de01-b8d5-4df4-883c-1a8429f18b56/img \
      --data_folder_flow /media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/flow \
      --n_workers 11 \
      --target_fphb 10 \
      --target_height 48 \
      --target_width 64 \
      --target_length 10
"

run_cmd="deepspeed --num_nodes 1 --num_gpus 2 main.py $@ ${options}"
echo ${run_cmd}
eval ${run_cmd}
set +x
