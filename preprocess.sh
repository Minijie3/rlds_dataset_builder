#!/bin/bash

python preprocess_split_aloha_data.py \
  --dataset_path /data3/embodied/galaxea/fold_the_towel \
  --out_base_dir /data3/embodied/galaxea/processed \
  --percent_val 0 \
  --img_resize_size 256