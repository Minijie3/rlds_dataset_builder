#!/bin/bash

python preprocess_split_aloha_data.py \
  --dataset_path /data3/embodied/galaxea/r1lite/put_the_duck_to_the_right_of_the_cup \
  --out_base_dir /data3/embodied/galaxea/r1lite/processed \
  --percent_val 0 \
  --img_resize_size 256