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

class RoboTwinBeatBlockHammer(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for RoboTwin beat_block_hammer dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release for RobotTwin beat_block_hammer dataset.',
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
                            doc='Head camera RGB observation.',
                        ),
                        'left_wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left wrist camera RGB observation.',
                        ),
                        'right_wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot state.',
                        ),
                        'tracks_image': tfds.features.Tensor(
                            shape=(784, 2),
                            dtype=np.float32,
                            doc='Optical flow tracks for head camera.',
                        ),
                        'tracks_left_wrist': tfds.features.Tensor(
                            shape=(784, 2),
                            dtype=np.float32,
                            doc='Optical flow tracks for left wrist camera.',
                        ),
                        'tracks_right_wrist': tfds.features.Tensor(
                            shape=(784, 2),
                            dtype=np.float32,
                            doc='Optical flow tracks for right wrist camera.',
                        ),
                        'sam_features_image': tfds.features.Tensor(
                            shape=(256, 256),  
                            dtype=np.float32,
                            doc='SAM features for head camera image.',
                        ),
                        'sam_features_left_wrist': tfds.features.Tensor(
                            shape=(256, 256),
                            dtype=np.float32,
                            doc='SAM features for left wrist camera image.',
                        ),
                        'sam_features_right_wrist': tfds.features.Tensor(
                            shape=(256, 256),
                            dtype=np.float32,
                            doc='SAM features for right wrist camera image.',
                        ),
                        'depth_features_image': tfds.features.Tensor(
                            shape=(224, 224),  
                            dtype=np.float32,
                            doc='Depth features for head camera image.',
                        ),
                        'depth_features_left_wrist': tfds.features.Tensor(
                            shape=(224, 224), 
                            dtype=np.float32,
                            doc='Depth features for left wrist camera image.',
                        ),
                        'depth_features_right_wrist': tfds.features.Tensor(
                            shape=(224, 224), 
                            dtype=np.float32,
                            doc='Depth features for right wrist camera image.',
                        ),
                        'laq_image_features': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for head camera image.',
                        ),
                        'laq_left_wrist_features': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for left wrist camera image.',
                        ),
                        'laq_right_wrist_features': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for right wrist camera image.',
                        ),
                        'laq_depth_features_image': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for head camera depth.',
                        ),
                        'laq_depth_features_left_wrist': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for left wrist camera depth.',
                        ),
                        'laq_depth_features_right_wrist': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for right wrist camera depth.',
                        ),
                        'laq_sam_features_image': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for head camera SAM.',
                        ),
                        'laq_sam_features_left_wrist': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for left wrist camera SAM.',
                        ),
                        'laq_sam_features_right_wrist': tfds.features.Tensor(
                            shape=(4, 1024),
                            dtype=np.float32,
                            doc='LAQ latent tokens for right wrist camera SAM.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot action.',
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
        train_files = glob.glob(
            "/data3/embodied/robotwin2.0/preprocessed/beat_block_hammer/train/*.hdf5"
        )

        print(f"[INFO] Found {len(train_files)} training files")

        return {
            "train": train_files
        }

    def _split_generators(self, dl_manager):
        """Define data splits."""
        split_paths = self._split_paths()
        return {split: self._generate_examples(paths=split_paths[split]) for split in split_paths}

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Yields episodes for list of data paths."""

        def _parse_example(episode_path):
            print(f"[INFO] Parsing file: {episode_path}")
            
            # load raw data
            with h5py.File(episode_path, "r") as F:
                # Check required keys
                required_keys = [
                    "/relative_action",
                    "/head_camera_image",
                    "/left_wrist_image",
                    "/right_wrist_image",
                    "/action",
                    "/seen",
                ]
                if not all(k in F for k in required_keys):
                    for key in required_keys:
                        if key not in F:
                            print(f"[ERROR] Missing key: {key} in {episode_path}")
                    print(f"[WARNING] Missing expected keys in {episode_path}, skipping")
                    return None
                
                T = F["/action"].shape[0]
                actions = F["/action"][1:].astype(np.float32)  # (T-1, 14)
                head_images = F["/head_camera_image"][:T-1].astype(np.uint8)
                left_wrist_images = F["/left_wrist_image"][:T-1].astype(np.uint8)
                right_wrist_images = F["/right_wrist_image"][:T-1].astype(np.uint8)
                states = F["/action"][:T-1].astype(np.float32)  # (T-1, 14)
                
                # Process seen instructions
                seen = [
                    s.decode("utf-8") if isinstance(s, bytes) else s for s in F["/seen"][()]
                ]
                T = T - 1  # Actual number of steps

                if not seen:
                    print(f"[ERROR] No 'seen' instructions found in {episode_path}")
                    return None

                if not (
                    head_images.shape[0] == left_wrist_images.shape[0] == 
                    right_wrist_images.shape[0] == T == states.shape[0]
                ):
                    print(f"[ERROR] Data length mismatch in {episode_path}")
                    return None

                # Use the first seen instruction as language instruction
                instruction = seen[0] if seen else ""

            # Extract tracks for all cameras
            tracks_image, _ = extract_tracks_for_episode(
                head_images, frame_gap=25, patch_size=8
            )
            tracks_left_wrist, _ = extract_tracks_for_episode(
                left_wrist_images, frame_gap=25, patch_size=8
            )
            tracks_right_wrist, _ = extract_tracks_for_episode(
                right_wrist_images, frame_gap=25, patch_size=8
            )
            
            # Extract SAM features for all cameras
            sam_features_image, _ = extract_sam_features(head_images)
            sam_features_left_wrist, _ = extract_sam_features(left_wrist_images)
            sam_features_right_wrist, _ = extract_sam_features(right_wrist_images)
            
            # Extract depth features for all cameras
            depth_features_image = extract_depth_features(head_images)
            depth_features_left_wrist = extract_depth_features(left_wrist_images)
            depth_features_right_wrist = extract_depth_features(right_wrist_images)
            
            # Extract LAQ features for all cameras and modalities
            laq_image_features = extract_laq_features(head_images, feature_type='image', offsets=25)
            laq_left_wrist_features = extract_laq_features(left_wrist_images, feature_type='image', offsets=25)
            laq_right_wrist_features = extract_laq_features(right_wrist_images, feature_type='image', offsets=25)
            
            laq_depth_features_image = extract_laq_features(depth_features_image, feature_type='depth', offsets=25)
            laq_depth_features_left_wrist = extract_laq_features(depth_features_left_wrist, feature_type='depth', offsets=25)
            laq_depth_features_right_wrist = extract_laq_features(depth_features_right_wrist, feature_type='depth', offsets=25)
            
            laq_sam_features_image = extract_laq_features(sam_features_image, feature_type='sam', offsets=25)
            laq_sam_features_left_wrist = extract_laq_features(sam_features_left_wrist, feature_type='sam', offsets=25)
            laq_sam_features_right_wrist = extract_laq_features(sam_features_right_wrist, feature_type='sam', offsets=25)

            # assemble episode with all features
            episode = []
            for i in range(actions.shape[0]):
                episode.append({
                    'observation': {
                        'image': head_images[i],
                        'left_wrist_image': left_wrist_images[i],
                        'right_wrist_image': right_wrist_images[i],
                        'state': np.asarray(states[i], dtype=np.float32),
                        'tracks_image': tracks_image[i].astype(np.float32),
                        'tracks_left_wrist': tracks_left_wrist[i].astype(np.float32),
                        'tracks_right_wrist': tracks_right_wrist[i].astype(np.float32),
                        'sam_features_image': sam_features_image[i].astype(np.float32),
                        'sam_features_left_wrist': sam_features_left_wrist[i].astype(np.float32),
                        'sam_features_right_wrist': sam_features_right_wrist[i].astype(np.float32),
                        'depth_features_image': depth_features_image[i].astype(np.float32),
                        'depth_features_left_wrist': depth_features_left_wrist[i].astype(np.float32),
                        'depth_features_right_wrist': depth_features_right_wrist[i].astype(np.float32),
                        'laq_image_features': laq_image_features[i].astype(np.float32),
                        'laq_left_wrist_features': laq_left_wrist_features[i].astype(np.float32),
                        'laq_right_wrist_features': laq_right_wrist_features[i].astype(np.float32),
                        'laq_depth_features_image': laq_depth_features_image[i].astype(np.float32),
                        'laq_depth_features_left_wrist': laq_depth_features_left_wrist[i].astype(np.float32),
                        'laq_depth_features_right_wrist': laq_depth_features_right_wrist[i].astype(np.float32),
                        'laq_sam_features_image': laq_sam_features_image[i].astype(np.float32),
                        'laq_sam_features_left_wrist': laq_sam_features_left_wrist[i].astype(np.float32),
                        'laq_sam_features_right_wrist': laq_sam_features_right_wrist[i].astype(np.float32),
                    },
                    'action': np.asarray(actions[i], dtype=np.float32),
                    'discount': 1.0,
                    'reward': float(i == (actions.shape[0] - 1)),
                    'is_first': i == 0,
                    'is_last': i == (actions.shape[0] - 1),
                    'is_terminal': i == (actions.shape[0] - 1),
                    'language_instruction': instruction,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            print(f"[INFO] Yielding {len(episode)} steps from {episode_path}")
            return episode_path, sample

        # Process each file
        for path in paths:
            result = _parse_example(path)
            if result is not None:
                yield result