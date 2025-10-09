from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py

import os
import requests
from unittest.mock import Mock

os.environ["TFDS_OFFLINE"] = "1"
os.environ["TFHUB_DISABLE_HTTP"] = "1"
os.environ["NO_GCE_CHECK"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
requests.get = Mock(side_effect=lambda *args, **kwargs: None)
requests.post = Mock(side_effect=lambda *args, **kwargs: None)
requests.head = Mock(side_effect=lambda *args, **kwargs: None)

import urllib3
urllib3.disable_warnings()
urllib3.PoolManager = Mock()

class AlohaFoldTheTshirtDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    NAME = 'aloha_fold_the_T-shirt'
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
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
                            doc='Robot joint state (7D left arm + 7D right arm).',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot arm action.',
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
                })
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/data3/embodied/galaxea/processed/fold_the_T-shirt/train/episode_*.hdf5")
        }

    def _split_generators(self, dl_manage):
        """Define data splits."""
        split_paths = self._split_paths()
        return {split: self._generate_examples(path=split_paths[split]) for split in split_paths}

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Yields episodes for list of data paths."""
        # the line below needs to be *inside* generate_examples so that each worker creates it's own model
        # creating one shared model outside this function would cause a deadlock

        def _parse_example(episode_path):
            print(f"Loading {episode_path}")
            # Load raw data
            with h5py.File(episode_path, "r") as F:
                actions = F["/action"][()]
                states = F["/observations/qpos"][()]
                images = F["/observations/images/cam_high"][()]  # Primary camera (top-down view)
                left_wrist_images = F["/observations/images/cam_left_wrist"][()]  # Left wrist camera
                right_wrist_images = F["/observations/images/cam_right_wrist"][()]  # Right wrist camera

            # Get language instruction
            # Assumes filepaths look like: "/PATH/TO/ALOHA/PREPROCESSED/DATASETS/<dataset_name>/train/episode_0.hdf5"
            raw_file_string = episode_path.split('/')[-3]
            command = " ".join(raw_file_string.split("_"))

            # Assemble episode: here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(actions.shape[0]):
                episode.append({
                    'observation': {
                        'image': images[i],
                        'left_wrist_image': left_wrist_images[i],
                        'right_wrist_image': right_wrist_images[i],
                        'state': np.asarray(states[i], np.float32),
                    },
                    'action': np.asarray(actions[i], dtype=np.float32),
                    'discount': 1.0,
                    'reward': float(i == (actions.shape[0] - 1)),
                    'is_first': i == 0,
                    'is_last': i == (actions.shape[0] - 1),
                    'is_terminal': i == (actions.shape[0] - 1),
                    'language_instruction': command,
                })

            # Create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # If you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # For smallish datasets, use single-thread parsing
        for sample in path:
            ret = _parse_example(sample)
            yield ret
