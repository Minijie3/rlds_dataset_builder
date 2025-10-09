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

from segment_anything.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.segment_anything.utils.transforms import ResizeLongestSide
from co_tracker.cotracker.predictor import CoTrackerPredictor

os.environ["TFDS_OFFLINE"] = "1"
os.environ["TFHUB_DISABLE_HTTP"] = "1"
os.environ["NO_GCE_CHECK"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
requests.get = Mock(side_effect=lambda *args, **kwargs: None)
requests.post = Mock(side_effect=lambda *args, **kwargs: None)
requests.head = Mock(side_effect=lambda *args, **kwargs: None)

_track_model = None
_sam_model = None
_depth_model = None

def get_points_on_a_grid(patch_size, image_size, device):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    H, W = image_size
    ph, pw = patch_size

    assert H % ph == 0 and W % pw == 0, "The patch size must divide the image dimensions"

    y_centers = np.arange(ph // 2, H, ph)
    x_centers = np.arange(pw // 2, W, pw)

    xv, yv = np.meshgrid(x_centers, y_centers)
    centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
    return torch.from_numpy(centers).to(device)

def extract_tracks_for_episode(images, frame_gap=8, patch_size=8):
    global _track_model
    
    if _track_model is None:
        checkpoint = "/data3/hj/rlds_dataset_builder/co_tracker/checkpoints/scaled_offline.pth"
        _track_model = CoTrackerPredictor(
            checkpoint=checkpoint,
            v2=False,
            offline=True,
            window_len=60
        )
        _track_model = _track_model.to('cuda').eval()
    
    video_frames = []
    for img in images:
        img_pil = Image.fromarray(img).resize((224, 224), resample=Image.BICUBIC)
        
        img_array = np.array(img_pil)[::-1, ::-1].copy()
        video_frames.append(img_array)
    
    video = np.stack(video_frames)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    
    if video_tensor.shape[0] < frame_gap + 1:
        video_segments = torch.zeros((0, 2, *video.shape[1:]))
    else:
        video_segments = video_tensor.unfold(0, frame_gap + 1, 1)
        video_segments = video_segments[..., [0, -1]]
        video_segments = video_segments.permute(0, 4, 1, 2, 3).contiguous()
    
    grid_pts = get_points_on_a_grid(patch_size, [224, 224], device='cpu').float().unsqueeze(0)
    grid_size = 224 // patch_size
    
    queries = torch.cat(
        [torch.ones_like(grid_pts[:, :, :1]) * 0, grid_pts],
        dim=2,
    ).repeat(video_segments.shape[0], 1, 1)
    
    video_segments = video_segments.to('cuda')
    queries = queries.to('cuda')
    
    batch_size = 32
    total_size = video_segments.shape[0]
    pred_tracks_list = []
    pred_visibility_list = []
    
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        pred_tracks_batch, pred_visibility_batch = _track_model(
            video_segments[start:end],
            queries=queries[start:end],
            grid_size=grid_size,
            backward_tracking=False,
        )
        pred_tracks_list.append(pred_tracks_batch)
        pred_visibility_list.append(pred_visibility_batch)
    
    pred_tracks = torch.cat(pred_tracks_list, dim=0)
    pred_visibility = torch.cat(pred_visibility_list, dim=0)
    
    pred_tracks_delta = (pred_tracks[:, 1:2, :, :] - pred_tracks[:, 0:1, :, :]).squeeze(1)
    
    tracks = pred_tracks_delta.cpu().numpy()  # [N, 784, 2]
    visibility = pred_visibility[:, 1, :].cpu().numpy()  # [N, 784]
    
    tracks_full = np.concatenate([tracks, np.zeros([frame_gap, 784, 2], dtype=np.float32)], axis=0)
    visibility_full = np.concatenate([visibility, np.zeros([frame_gap, 784], dtype=np.float32)], axis=0)
    
    return tracks_full, visibility_full

def extract_sam_features(images):
    global _sam_model, _mask_generator
    masks = None
    
    if _sam_model is None:
        checkpoint = "/data3/hj/rlds_dataset_builder/segment_anything/ckpts/sam_vit_b_01ec64.pth"
        model_type = '_'.join(os.path.basename(checkpoint).split('_')[1:3])
        _sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        _sam_model = _sam_model.to('cuda').eval()

        # === Only for debug ===
        # _mask_generator = SamAutomaticMaskGenerator(_sam_model)
    
    transform = ResizeLongestSide(_sam_model.image_encoder.img_size)
    processed_images = []
    
    for img in images:
        img_processed = np.array(img[::-1, ::-1]).copy()
        img_processed = transform.apply_image(img_processed)
        img_tensor = torch.as_tensor(img_processed).permute(2, 0, 1).contiguous()
        processed_images.append(img_tensor)
    
    batch_tensor = torch.stack(processed_images).cpu()
    
    batch_size = 4
    total_size = batch_tensor.shape[0]
    features_list = []
    
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch_input = batch_tensor[start:end].float().to('cuda')
        
        with torch.no_grad():
            features_batch = _sam_model.image_encoder(batch_input) # [B, C, H, W]
            features_batch = torch.nn.functional.avg_pool2d(features_batch, kernel_size=4, stride=4, padding=0)
            features_batch = features_batch.flatten(start_dim=-2)
            features_list.append(features_batch)
    
    features = torch.cat(features_list, dim=0)

    # === Only for debug ===
    # masks = _mask_generator.generate(images[0][::-1, ::-1])
    
    return features.to(torch.float32).cpu().numpy(), masks

def extract_depth_features(images):
    global _depth_model
    
    if _depth_model is None:
        from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitb'
        
        _depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_ckpt = torch.load(f'/data3/hj/rlds_dataset_builder/Depth_Anything_V2/depth_anything_v2_{encoder}.pth', map_location='cpu')
        _depth_model.load_state_dict(depth_ckpt)
        _depth_model = _depth_model.to("cuda").eval()
    
    transform = ResizeLongestSide(_depth_model.pretrained.patch_embed.img_size[0])
    processed_images = []
    
    for img in images:
        img_processed = np.array(img[::-1, ::-1]).copy()
        img_processed = transform.apply_image(img_processed)
        img_tensor = torch.as_tensor(img_processed).permute(2, 0, 1).contiguous()
        processed_images.append(img_tensor)
    
    batch_tensor = torch.stack(processed_images).to('cuda')
    
    batch_size = 128
    total_size = batch_tensor.shape[0]
    features_list = []
    
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch_input = batch_tensor[start:end].permute(0, 2, 3, 1)  # [B, H, W, C]
        
        with torch.no_grad():
            for i in range(batch_input.shape[0]):
                img_np = batch_input[i].cpu().numpy()
                feature = _depth_model.infer_image(img_np, _depth_model.pretrained.patch_embed.img_size[0])
                features_list.append(torch.tensor(feature))
    
    features = torch.stack(features_list, dim=0)
    
    return features.to(torch.float32).cpu().numpy()  # [T, ...]

class LIBERO10(tfds.core.GeneratorBasedBuilder):
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
                            shape=(518, 518),  
                            dtype=np.float32,
                            doc='Depth features for main camera image.',
                        ),
                        'depth_features_wrist': tfds.features.Tensor(
                            shape=(518, 518),
                            dtype=np.float32,
                            doc='Depth features for wrist camera image.',
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
            "train": glob.glob("/data3/embodied/modified_libero_rlds_feats/libero_10_no_noops/*.hdf5"),
        }

    def _split_generators(self, dl_manager):
        """Define data splits."""
        split_paths = self._split_paths()
        return {split: self._generate_examples(paths=split_paths[split]) for split in split_paths}

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Yields episodes for list of data paths."""
        # the line below needs to be *inside* generate_examples so that each worker creates it's own model
        # creating one shared model outside this function would cause a deadlock

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

            # if you want to skip an example for whatever reason, simply return None
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