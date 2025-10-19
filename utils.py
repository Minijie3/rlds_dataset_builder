import os
import sys
import glob
import requests
from unittest.mock import Mock
from typing import Iterator, Tuple, Any
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms as T

from segment_anything.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.segment_anything.utils.transforms import ResizeLongestSide
from co_tracker.cotracker.predictor import CoTrackerPredictor
from LAPA.laq.laq_model.latent_action_quantization import LatentActionQuantization

# Global model variables
_track_model = None
_sam_model = None
_depth_model = None
_laq_image_model = None
_laq_depth_model = None
_laq_sam_model = None

def get_points_on_a_grid(patch_size, image_size, device):
    """Generate grid points for tracking."""
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
    """Extract optical flow tracks from image sequence."""
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
    """Extract SAM features from images."""
    global _sam_model, _mask_generator
    masks = None
    
    if _sam_model is None:
        checkpoint = "/data3/hj/rlds_dataset_builder/segment_anything/ckpts/sam_vit_b_01ec64.pth"
        model_type = '_'.join(os.path.basename(checkpoint).split('_')[1:3])
        _sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        _sam_model = _sam_model.to('cuda').eval()

        # Only for debug
        _mask_generator = SamAutomaticMaskGenerator(_sam_model)
    
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

    # Only for debug
    masks = _mask_generator.generate(images[0][::-1, ::-1])
    
    return features.to(torch.float32).cpu().numpy(), masks

def extract_depth_features(images):
    """Extract depth features from images."""
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
    
    batch_size = 32
    total_size = batch_tensor.shape[0]
    features_list = []
    
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch_input = batch_tensor[start:end].permute(0, 2, 3, 1)  # [B, H, W, C]
        
        with torch.no_grad():
            for i in range(batch_input.shape[0]):
                img_np = batch_input[i].cpu().numpy()
                feature = _depth_model.infer_image(img_np, _depth_model.pretrained.patch_embed.img_size[0])
                feature_resized = cv2.resize(feature, (224, 224), interpolation=cv2.INTER_LINEAR)
                features_list.append(torch.tensor(feature_resized))
    
    features = torch.stack(features_list, dim=0)
    
    return features.to(torch.float32).cpu().numpy()  # [T, ...]

def extract_laq_features(images, feature_type='image', offsets=8):
    global _laq_image_model, _laq_depth_model, _laq_sam_model
    
    # Determine which model to use based on feature type
    if feature_type == 'image':
        if _laq_image_model is None:
            _laq_image_model = LatentActionQuantization(
                dim = 1024,
                quant_dim=32,
                codebook_size = 8,
                image_size = 256,
                patch_size = 32,
                spatial_depth = 8, #8
                temporal_depth = 8, #8
                dim_head = 64,
                heads = 16,
                code_seq_len=4,
            )
            # Load checkpoint for image model
            checkpoint_path = "/data3/hj/rlds_dataset_builder/LAPA/laq/results/vae.100000.pt"  # Update path
            _laq_image_model.load(checkpoint_path)
            _laq_image_model = _laq_image_model.to('cuda').eval()
        model = _laq_image_model
        
    elif feature_type == 'depth':
        if _laq_depth_model is None:
            _laq_depth_model = LatentActionQuantization(
                dim = 1024,
                quant_dim=32,
                codebook_size = 8,
                image_size = 224,
                patch_size = 32,
                spatial_depth = 8, #8
                temporal_depth = 8, #8
                dim_head = 64,
                heads = 16,
                code_seq_len=4,
            )
            # Load checkpoint for depth model
            checkpoint_path = "/data3/hj/rlds_dataset_builder/LAPA/laq/results_depth/vae.100000.pt"  # Update path
            _laq_depth_model.load(checkpoint_path)
            _laq_depth_model = _laq_depth_model.to('cuda').eval()
        model = _laq_depth_model
        
    elif feature_type == 'sam':
        if _laq_sam_model is None:
            _laq_sam_model = LatentActionQuantization(
                dim = 1024,
                quant_dim=32,
                codebook_size = 8,
                image_size = 256,
                patch_size = 32,
                spatial_depth = 8, #8
                temporal_depth = 8, #8
                dim_head = 64,
                heads = 16,
                code_seq_len=4,
            )
            # Load checkpoint for SAM model
            checkpoint_path = "/data3/hj/rlds_dataset_builder/LAPA/laq/results_sam/vae.100000.pt"  # Update path
            _laq_sam_model.load(checkpoint_path)
            _laq_sam_model = _laq_sam_model.to('cuda').eval()
        model = _laq_sam_model
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    # Process images
    if feature_type == 'image':
        transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize(256),
                T.ToTensor(),
            ])
    else:
        transform = T.Compose([
                T.Lambda(lambda x: torch.from_numpy(x).float()),
                T.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),  # Add channel dimension if 2D
            ])

    processed_images = []
    for img in images:
        if feature_type == 'image':
            img_flipped = img[::-1, ::-1]
            img_pil = Image.fromarray(img_flipped)
            img_processed = transform(img_pil)
        else:
            img_processed = transform(img)
        processed_images.append(img_processed)
    
    # Convert processed images to tensor [T, C, H, W]
    video_tensor = torch.stack(processed_images).float()
    
    # Create frame pairs using unfold similar to track extraction function
    if video_tensor.shape[0] < offsets + 1:
        video_segments = torch.zeros((0, video_tensor.shape[1], 2, *video_tensor.shape[2:]))
    else:
        video_segments = video_tensor.unfold(0, offsets + 1, 1)
        video_segments = video_segments[..., [0, -1]]
        video_segments = video_segments.permute(0, 1, 4, 2, 3).contiguous()  # [N, C, 2, H, W]
    
    # Convert to tensor and process in batches
    batch_size = 32
    total_size = video_segments.shape[0]
    all_tokens = []
    seq_len = None
    
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch_segments = video_segments[start:end]

        batch_tensor = batch_segments.to('cuda')
        
        with torch.no_grad():
            # Use obtain_tokens method to get the latent tokens
            tokens = model.obtain_tokens(batch_tensor) # [B, 4, 1024]
            seq_len = tokens.shape[1]
            all_tokens.append(tokens.cpu().numpy())

    laq_tokens_partial = np.concatenate(all_tokens, axis=0)
    
    # Pad zeros for the frames that couldn't form pairs (last 'offsets' frames)
    total_frames = len(images)
    feature_dim = laq_tokens_partial.shape[2]
    laq_tokens = np.zeros((total_frames, seq_len, feature_dim), dtype=np.float32)
    valid_length = laq_tokens_partial.shape[0]  # = T - offsets
    laq_tokens[:valid_length] = laq_tokens_partial
    
    return laq_tokens