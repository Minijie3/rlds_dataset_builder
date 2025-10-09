import os
import h5py
import zarr
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import numcodecs

UMI_IMAGE_SIZE = (224, 224)

def convert_aloha_to_umi_zarr(hdf5_dir: str, output_zarr_path: str):
    episode_files = sorted([
        os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) 
        if f.endswith('.hdf5') and f.startswith('episode_')
    ])
    
    if not episode_files:
        raise ValueError(f"[ERR] No HDF5 files found in directory: {hdf5_dir}")

    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    total_steps = 0
    episode_ends = []
    
    print("Precomputing dataset dimensions...")
    for file_path in episode_files:
        with h5py.File(file_path, 'r') as f:
            total_steps += f['action'].shape[0]
            episode_ends.append(total_steps)

    data_group.zeros(
        'camera0_rgb',
        shape=(total_steps, *UMI_IMAGE_SIZE, 3),
        chunks=(100, None, None, None),
        dtype='uint8'
    )
    data_group.zeros(
        'robot0_eef_pos',
        shape=(total_steps, 3),
        chunks=(500, None),
        dtype='float32'
    )
    data_group.zeros(
        'robot0_eef_rot_axis_angle',
        shape=(total_steps, 3),
        chunks=(500, None),
        dtype='float32'
    )
    data_group.zeros(
        'robot0_gripper_width',
        shape=(total_steps, 1),
        chunks=(500, None),
        dtype='float32'
    )
    data_group.zeros(
        'robot0_demo_start_pose',
        shape=(total_steps, 6),
        chunks=(500, None),
        dtype='float64'
    )
    data_group.zeros(
        'robot0_demo_end_pose',
        shape=(total_steps, 6),
        chunks=(500, None),
        dtype='float64'
    )
    meta_group.array('episode_ends', np.array(episode_ends, dtype=np.int64))

    current_idx = 0
    print(f"Converting {len(episode_files)} episodes to UMI Zarr format...")
    
    for file_path in tqdm(episode_files):
        with h5py.File(file_path, 'r') as f:
            num_steps = f['action'].shape[0]
            end_idx = current_idx + num_steps

            data_group['camera0_rgb'][current_idx:end_idx] = [
                img for img in f['observations/images/cam_left_wrist'][:]
            ]

            qpos = f['observations/qpos'][:, :7]

            data_group['robot0_eef_pos'][current_idx:end_idx] = qpos[:, :3]
            data_group['robot0_eef_rot_axis_angle'][current_idx:end_idx] = qpos[:, 3:6]
            data_group['robot0_gripper_width'][current_idx:end_idx] = qpos[:, 6:7]

            start_pose = np.concatenate([
                qpos[0, :3],
                qpos[0, 3:6]
            ])
            end_pose = np.concatenate([
                qpos[-1, :3],
                qpos[-1, 3:6]
            ])
            start_pose = np.tile(start_pose, (num_steps, 1))
            end_pose = np.tile(end_pose, (num_steps, 1))

            data_group['robot0_demo_start_pose'][current_idx:end_idx] = start_pose
            data_group['robot0_demo_end_pose'][current_idx:end_idx] = end_pose
            
            current_idx = end_idx
    
    print(f"Conversion complete. UMI Zarr dataset saved to: {output_zarr_path}")
    print(f"Total timesteps: {total_steps}, Episodes: {len(episode_files)}")
    print("Dataset structure:")
    print(root.tree())

if __name__ == "__main__":
    convert_aloha_to_umi_zarr(
        hdf5_dir="/data3/zyj/unified_video_action_origin/data/processed/umi_put_the_cube_into_the_plate_noend/train",
        output_zarr_path="/data3/zyj/unified_video_action_origin/data/aloha_cube/cube_put_0.zarr"
    )

