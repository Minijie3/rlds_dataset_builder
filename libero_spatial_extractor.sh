# !/bin/bash

export CUDA_VISIBLE_DEVICES=6  
cd /data3/hj/rlds_dataset_builder/LIBERO_Spatial
tfds build --data_dir /data3/embodied/modified_libero_rlds_feats/tensorflow_datasets