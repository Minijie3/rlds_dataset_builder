import os
import sys
import glob
import requests
from unittest.mock import Mock
from typing import Iterator, Tuple, Any

import cv2
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image

sys.path.insert(0, "/data3/hj/rlds_dataset_builder")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils import (
    extract_tracks_for_episode, 
    extract_sam_features, 
    extract_depth_features,
    extract_laq_features
)

os.environ["TFDS_OFFLINE"] = "1"
os.environ["TFHUB_DISABLE_HTTP"] = "1"
os.environ["NO_GCE_CHECK"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
requests.get = Mock(side_effect=lambda *args, **kwargs: None)
requests.post = Mock(side_effect=lambda *args, **kwargs: None)
requests.head = Mock(side_effect=lambda *args, **kwargs: None)

class LIBEROGoal(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata with all features."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        ),
                        'tracks_image': tfds.features.Tensor(
                            shape=(784, 2),
                            dtype=np.float32,
                            doc='Optical flow tracks for main camera.',
                        ),
                        'visibility_image': tfds.features.Tensor(
                            shape=(784,),
                            dtype=np.float32,
                            doc='Visibility mask for main camera tracks.',
                        ),
                        'tracks_wrist': tfds.features.Tensor(
                            shape=(784, 2),
                            dtype=np.float32,
                            doc='Optical flow tracks for wrist camera.',
                        ),
                        'visibility_wrist': tfds.features.Tensor(
                            shape=(784,),
                            dtype=np.float32,
                            doc='Visibility mask for wrist camera tracks.',
                        ),
                        'sam_features_image': tfds.features.Tensor(
                            shape=(256, 256),  
                            dtype=np.float32,
                            doc='SAM features for main camera image.',
                        ),
                        'sam_features_wrist': tfds.features.Tensor(
                            shape=(256, 256),
                            dtype=np.float32,
                            doc='SAM features for wrist camera image.',
                        ),
                        'depth_features_image': tfds.features.Tensor(
                            shape=(224, 224),  
                            dtype=np.float32,
                            doc='Depth features for main camera image.',
                        ),
                        'depth_features_wrist': tfds.features.Tensor(
                            shape=(224, 224), 
                            dtype=np.float32,
                            doc='Depth features for wrist camera image.',
                        ),
                        'laq_image_features': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for main camera image.',
                        ),
                        'laq_wrist_features': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for wrist camera image.',
                        ),
                        'laq_depth_features_image': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for main camera depth.',
                        ),
                        'laq_depth_features_wrist': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for wrist camera depth.',
                        ),
                        'laq_sam_features_image': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for main camera SAM.',
                        ),
                        'laq_sam_features_wrist': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for wrist camera SAM.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/data3/embodied/modified_libero_rlds_feats/libero_goal_no_noops/*.hdf5"),
        }

    def _split_generators(self, dl_manager):
        """Define data splits."""
        split_paths = self._split_paths()
        return {split: self._generate_examples(paths=split_paths[split]) for split in split_paths}

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Yields episodes for list of data paths."""

        def _parse_example(episode_path, demo_id):
            # load raw data
            with h5py.File(episode_path, "r") as F:
                if f"demo_{demo_id}" not in F['data'].keys():
                    return None  # skip episode if the demo doesn't exist
                
                actions = F['data'][f"demo_{demo_id}"]["actions"][()]
                states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
                gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
                joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
                images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
                wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]

            tracks_image, visibility_image = extract_tracks_for_episode(
                images, frame_gap=8, patch_size=8
            )
            tracks_wrist, visibility_wrist = extract_tracks_for_episode(
                wrist_images, frame_gap=8, patch_size=8
            )
            
            sam_features_image, _ = extract_sam_features(images)
            sam_features_wrist, _ = extract_sam_features(wrist_images)
            
            depth_features_image = extract_depth_features(images)
            depth_features_wrist = extract_depth_features(wrist_images)
            
            laq_image_features = extract_laq_features(images, feature_type='image')
            laq_wrist_features = extract_laq_features(wrist_images, feature_type='image')
            laq_depth_features_image = extract_laq_features(depth_features_image, feature_type='depth')
            laq_depth_features_wrist = extract_laq_features(depth_features_wrist, feature_type='depth')
            laq_sam_features_image = extract_laq_features(sam_features_image, feature_type='sam')
            laq_sam_features_wrist = extract_laq_features(sam_features_wrist, feature_type='sam')

            # compute language instruction
            raw_file_string = os.path.basename(episode_path).split('/')[-1]
            words = raw_file_string[:-10].split("_")
            command = ''
            for w in words:
                if "SCENE" in w:
                    command = ''
                    continue
                command = command + w + ' '
            command = command[:-1]

            # assemble episode with all features
            episode = []
            for i in range(actions.shape[0]):
                episode.append({
                    'observation': {
                        'image': images[i][::-1, ::-1],
                        'wrist_image': wrist_images[i][::-1, ::-1],
                        'state': np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                        'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                        'tracks_image': tracks_image[i].astype(np.float32),
                        'visibility_image': visibility_image[i].astype(np.float32),
                        'tracks_wrist': tracks_wrist[i].astype(np.float32),
                        'visibility_wrist': visibility_wrist[i].astype(np.float32),
                        'sam_features_image': sam_features_image[i].astype(np.float32),
                        'sam_features_wrist': sam_features_wrist[i].astype(np.float32),
                        'depth_features_image': depth_features_image[i].astype(np.float32),
                        'depth_features_wrist': depth_features_wrist[i].astype(np.float32),
                        'laq_image_features': laq_image_features[i].astype(np.float32),
                        'laq_wrist_features': laq_wrist_features[i].astype(np.float32),
                        'laq_depth_features_image': laq_depth_features_image[i].astype(np.float32),
                        'laq_depth_features_wrist': laq_depth_features_wrist[i].astype(np.float32),
                        'laq_sam_features_image': laq_sam_features_image[i].astype(np.float32),
                        'laq_sam_features_wrist': laq_sam_features_wrist[i].astype(np.float32),
                    },
                    'action': np.asarray(actions[i], dtype=np.float32),
                    'discount': 1.0,
                    'reward': float(i == (actions.shape[0] - 1)),
                    'is_first': i == 0,
                    'is_last': i == (actions.shape[0] - 1),
                    'is_terminal': i == (actions.shape[0] - 1),
                    'language_instruction': command,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return episode_path + f"_{demo_id}", sample

        # for smallish datasets, use single-thread parsing
        for sample in paths:
            with h5py.File(sample, "r") as F:
                n_demos = len(F['data'])
            idx = 0
            cnt = 0
            while cnt < n_demos:
                ret = _parse_example(sample, idx)
                if ret is not None:
                    cnt += 1
                    idx += 1
                    yield ret
                else:
                    idx += 1
                    print(f"Skipping demo {idx} in {sample} since it does not exist.")