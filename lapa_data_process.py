import os
import glob
import h5py
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Dict, Any

from segment_anything.segment_anything import sam_model_registry
from segment_anything.segment_anything.utils.transforms import ResizeLongestSide
from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

_sam_model = None
_depth_model = None

def extract_sam_features(images: List[np.ndarray]) -> np.ndarray:
    global _sam_model
    
    if _sam_model is None:
        checkpoint = "/data3/hj/rlds_dataset_builder/segment_anything/ckpts/sam_vit_b_01ec64.pth"
        model_type = '_'.join(os.path.basename(checkpoint).split('_')[1:3])
        _sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        _sam_model = _sam_model.to('cuda').eval()
    
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
            features_batch = _sam_model.image_encoder(batch_input)
            features_batch = torch.nn.functional.avg_pool2d(features_batch, kernel_size=4, stride=4, padding=0)
            features_batch = features_batch.flatten(start_dim=-2)
            features_list.append(features_batch)
    
    features = torch.cat(features_list, dim=0)
    return features.to(torch.float32).cpu().numpy()

def extract_depth_features(images: List[np.ndarray]) -> np.ndarray:
    global _depth_model
    
    if _depth_model is None:
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
        batch_input = batch_tensor[start:end].permute(0, 2, 3, 1)
        
        with torch.no_grad():
            for i in range(batch_input.shape[0]):
                img_np = batch_input[i].cpu().numpy()
                feature = _depth_model.infer_image(img_np, _depth_model.pretrained.patch_embed.img_size[0])
                
                # Resize depth to 256*256
                feature_resized = cv2.resize(feature, (256, 256), interpolation=cv2.INTER_LINEAR)
                features_list.append(torch.tensor(feature_resized))
    
    features = torch.stack(features_list, dim=0)
    return features.to(torch.float32).cpu().numpy()

def process_libero_dataset(input_base_path: str, output_base_path: str):
    hdf5_files = glob.glob(os.path.join(input_base_path, "*.hdf5"))
    print(f"Found {len(hdf5_files)} hdf5 files")
    
    video_counter = 0
    
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file}...")
        
        with h5py.File(hdf5_file, "r") as F:
            # Get all demos in this file
            demo_keys = [key for key in F['data'].keys() if key.startswith('demo_')]
            print(f"Found {len(demo_keys)} demos in {os.path.basename(hdf5_file)}")
            
            for demo_key in demo_keys:
                # Create output directory for this video
                video_dir = os.path.join(output_base_path, f"video_{video_counter:04d}")
                os.makedirs(video_dir, exist_ok=True)
                
                # Extract data
                demo_group = F['data'][demo_key]
                images = demo_group['obs']['agentview_rgb'][()]  # Main camera
                wrist_images = demo_group['obs']['eye_in_hand_rgb'][()]  # Wrist camera
                
                num_frames = images.shape[0]
                print(f"Processing demo {demo_key} with {num_frames} frames...")
                
                # Extract features in batches to save memory
                batch_size = 32
                
                for start_idx in range(0, num_frames, batch_size):
                    end_idx = min(start_idx + batch_size, num_frames)
                    
                    # Process main camera images
                    batch_images = [images[i] for i in range(start_idx, end_idx)]
                    
                    # Extract SAM features for main camera
                    sam_features = extract_sam_features(batch_images)
                    
                    # Extract depth features for main camera
                    depth_features = extract_depth_features(batch_images)
                    
                    # Process wrist camera images
                    batch_wrist_images = [wrist_images[i] for i in range(start_idx, end_idx)]
                    
                    # Extract SAM features for wrist camera
                    sam_features_wrist = extract_sam_features(batch_wrist_images)
                    
                    # Extract depth features for wrist camera
                    depth_features_wrist = extract_depth_features(batch_wrist_images)
                    
                    # Save each frame in the batch
                    for batch_idx, frame_idx in enumerate(range(start_idx, end_idx)):
                        frame_num = frame_idx
                        
                        # Main camera data
                        image_main = batch_images[batch_idx][::-1, ::-1]  # Apply flip
                        image_main_pil = Image.fromarray(image_main)
                        image_main_pil.save(os.path.join(video_dir, f"image_{frame_num:04d}.jpg"))
                        
                        np.save(os.path.join(video_dir, f"depth_{frame_num:04d}.npy"), 
                               depth_features[batch_idx])
                        np.save(os.path.join(video_dir, f"sam_{frame_num:04d}.npy"), 
                               sam_features[batch_idx])
                        
                        # Wrist camera data
                        image_wrist = batch_wrist_images[batch_idx][::-1, ::-1]  # Apply flip
                        image_wrist_pil = Image.fromarray(image_wrist)
                        image_wrist_pil.save(os.path.join(video_dir, f"image_wrist_{frame_num:04d}.jpg"))
                        
                        np.save(os.path.join(video_dir, f"depth_wrist_{frame_num:04d}.npy"), 
                               depth_features_wrist[batch_idx])
                        np.save(os.path.join(video_dir, f"sam_wrist_{frame_num:04d}.npy"), 
                               sam_features_wrist[batch_idx])
                
                print(f"Completed video_{video_counter:04d} with {num_frames} frames")
                video_counter += 1
    
    print(f"Processing complete! Total videos processed: {video_counter}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LIBERO dataset and extract images with features")
    parser.add_argument("--input_base_path", type=str, required=True, help="Input path to LIBERO dataset")
    parser.add_argument("--output_base_path", type=str, required=True, help="Output path for processed dataset")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_base_path, exist_ok=True)
    
    # Process the dataset
    process_libero_dataset(args.input_base_path, args.output_base_path)