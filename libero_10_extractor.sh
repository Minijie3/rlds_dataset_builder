# !/bin/bash

export CUDA_VISIBLE_DEVICES=7  
cd /data3/hj/rlds_dataset_builder/LIBERO_10
tfds build --data_dir /data3/embodied/modified_libero_rlds_feats/tensorflow_datasets