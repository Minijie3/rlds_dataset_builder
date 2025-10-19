import os
import h5py
import numpy as np
from PIL import Image
import glob

def process_episode(episode_path, output_path, episode_index):
    """
    Process the data of a single episode and convert it to HDF5 format
    
    Args:
        episode_path: Path to the original episode data
        output_path: Path to the output HDF5 file
        episode_index: Episode index (starting from 0)
    """
    
    # 1. Read joint data files
    joint_files = sorted(glob.glob(os.path.join(episode_path, "joint_data_*.txt")))
    num_frames = len(joint_files)
    
    print(f"Converting episode {episode_index} with {num_frames} frames...")
    
    # 2. Initialize data arrays
    action_data = np.zeros((num_frames, 14), dtype=np.float32)
    base_action_data = np.zeros((num_frames, 2), dtype=np.float32)
    effort_data = np.zeros((num_frames, 14), dtype=np.float32)
    qpos_data = np.zeros((num_frames, 14), dtype=np.float32)
    qvel_data = np.zeros((num_frames, 14), dtype=np.float32)
    
    # 3. Process joint data
    for i, joint_file in enumerate(joint_files):
        with open(joint_file, 'r') as f:
            lines = f.readlines()
            # Skip comment lines and read data lines
            for line in lines:
                if line.startswith('#'):
                    continue
                # Parse 14 joint position data
                joint_values = [float(x) for x in line.strip().split(',')]
                
                # Process qpos data
                qpos_data[i, :] = joint_values
                
                # Process action data
                action_data[i, :] = joint_values[:] 
    
    # 4. Process image data
    # Define camera path mapping
    camera_paths = {
        'cam_high': os.path.join(episode_path, "Replicator", "rgb"),
        'cam_left_wrist': os.path.join(episode_path, "Replicator_01", "rgb"), 
        'cam_right_wrist': os.path.join(episode_path, "Replicator_02", "rgb")
    }
    
    # Initialize image data dictionary
    image_data = {}
    
    # Read image data from each camera
    for cam_name, cam_path in camera_paths.items():
        if not os.path.exists(cam_path):
            print(f"Warning: Camera path {cam_path} does not exist, skipping...")
            continue
            
        image_files = sorted(glob.glob(os.path.join(cam_path, "rgb_*.png")))
        
        if len(image_files) != num_frames:
            print(f"Warning: Number of images ({len(image_files)}) doesn't match joint data ({num_frames}) for {cam_name}")
            # Use the smaller number
            actual_frames = min(len(image_files), num_frames)
        else:
            actual_frames = num_frames
        
        # Read the first image to get size information
        if actual_frames > 0:
            first_img = Image.open(image_files[0])
            # Convert RGBA to RGB if needed
            if first_img.mode == 'RGBA':
                first_img = first_img.convert('RGB')
            img_array = np.array(first_img)
            h, w, c = img_array.shape
            print(f"{cam_name} image shape: {h}x{w}x{c}")
            
            # Initialize the image array for this camera
            cam_images = np.zeros((actual_frames, h, w, c), dtype=np.uint8)
            
            # Read all images
            for j in range(actual_frames):
                img = Image.open(image_files[j])
                # Convert RGBA to RGB if needed
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                cam_images[j] = np.array(img)
            
            image_data[cam_name] = cam_images
        else:
            print(f"Warning: No images found for {cam_name}")
    
    # 5. Create an HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create top-level datasets
        f.create_dataset('action', data=action_data)
        f.create_dataset('base_action', data=base_action_data)
        
        # Create an observations group
        obs_group = f.create_group('observations')
        obs_group.create_dataset('effort', data=effort_data)
        obs_group.create_dataset('qpos', data=qpos_data)
        obs_group.create_dataset('qvel', data=qvel_data)
        
        # Create an images group
        images_group = obs_group.create_group('images')
        for cam_name, cam_images in image_data.items():
            images_group.create_dataset(cam_name, data=cam_images)
    
    print(f"Successfully created {output_path}")

def main():
    # Configure paths
    input_base_dir = "/data3/embodied/galaxea/r1lite/put_the_duck_to_the_right_of_the_cup/data_issac"
    output_base_dir = "/data3/embodied/galaxea/r1lite/put_the_duck_to_the_right_of_the_cup"
    
    # Ensure the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process all episodes (01-50)
    for i in range(1, 51):
        episode_dir = f"{i}"
        episode_path = os.path.join(input_base_dir, episode_dir)
        output_filename = f"episode_{i-1}.hdf5"  # Index starting from 0
        output_path = os.path.join(output_base_dir, output_filename)
        
        if os.path.exists(episode_path):
            try:
                print(f"Processing {episode_path}...")
                process_episode(episode_path, output_path, i-1)
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
        else:
            print(f"Episode path {episode_path} does not exist, skipping...")

if __name__ == "__main__":
    main()