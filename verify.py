import tensorflow_datasets as tfds
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_features_tfrecord(results, save_path=None):
    """Visualize the extracted features from TFRecord - strictly following the original format"""
    
    image = results['image']
    tracks = results['tracks']
    visibility = results['visibility']
    sam_features = results['sam_features']
    depth_features = results['depth_features']
    instruction = results['instruction']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Track visualization (show tracks in the last frame)
    # Reshape tracks to 28x28x2 for visualization
    tracks_reshaped = tracks.reshape(28, 28, 2)
    axes[0, 1].imshow(Image.fromarray(image).resize((224, 224), resample=Image.BICUBIC))
    # Draw track points on the image - exactly like original
    for i in range(28):
        for j in range(28):
            dx, dy = tracks_reshaped[i, j]
            x = j * 8 + 4  # Calculate the original position (center of the grid)
            y = i * 8 + 4
            if i % 2 == 0 and j % 2 == 0:  # Draw every other point to avoid overcrowding
                axes[0, 1].arrow(x, y, dx * 5, dy * 5, head_width=3, head_length=2, fc='red', ec='red')
    axes[0, 1].set_title('Optical Flow Tracks')
    axes[0, 1].axis('off')
    
    # Visibility heatmap
    last_frame_visibility = visibility.reshape(28, 28)
    im = axes[0, 2].imshow(last_frame_visibility, cmap='hot')
    axes[0, 2].set_title('Visibility Heatmap')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # SAM feature visualization (take the average if needed, but TFRecord SAM is already 2D)
    sam_avg = sam_features.mean(axis=0).reshape(16, 16)
    im = axes[1, 0].imshow(sam_avg, cmap='viridis')
    axes[1, 0].set_title('SAM Features (avg)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Empty subplot (removed SAM mask visualization as requested)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('SAM Masks (Not Visualized)')
    
    # Depth feature visualization
    if len(depth_features.shape) == 2:
        depth_vis = axes[1, 2].imshow(depth_features, cmap='gray')
    else:
        depth_avg = depth_features.mean(axis=0)
        depth_vis = axes[1, 2].imshow(depth_avg, cmap='gray')
    axes[1, 2].set_title('Depth Features')
    axes[1, 2].axis('off')
    plt.colorbar(depth_vis, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Statistics text - exactly like original format
    stats_text = f"""
    Feature Statistics:
    - Tracks shape: {tracks.shape}
    - Visibility shape: {visibility.shape}
    - SAM features shape: {sam_features.shape}
    - Depth features shape: {depth_features.shape}
    
    Value Ranges:
    - Tracks: [{tracks.min():.3f}, {tracks.max():.3f}]
    - Visibility: [{visibility.min():.3f}, {visibility.max():.3f}]
    - SAM: [{sam_features.min():.3f}, {sam_features.max():.3f}]
    - Depth: [{depth_features.min():.3f}, {depth_features.max():.3f}]
    
    Instruction: {instruction}
    """
    
    plt.figtext(0.35, 0.15, stats_text, fontsize=10, fontfamily='monospace')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization results saved to: {save_path}")
    
    plt.show()

def main():
    # Load dataset from directory
    data_dir = "/data3/embodied/modified_libero_rlds_feats/tensorflow_datasets/libero_object_no_noops/1.0.0"

    # Build dataset from directory
    ds = tfds.builder_from_directory(data_dir).as_dataset(split='train')
    
    # Get one episode
    episode = next(iter(ds.take(1)))
    
    # Convert to numpy and get steps
    steps = list(tfds.as_numpy(episode['steps']))
    
    # Randomly select one step
    random_step = random.choice(steps)
    
    print(f"Selected step from episode with {len(steps)} steps")
    
    # Extract features from TFRecord
    image_data = random_step['observation']['image']
    track_data = random_step['observation']['tracks_image']
    visibility_data = random_step['observation']['visibility_image']
    sam_data = random_step['observation']['sam_features_image']
    depth_data = random_step['observation']['depth_features_image']
    instruction = random_step['language_instruction']
    action = random_step['action']
    
    # Convert image to RGB for visualization
    image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    
    print("Feature shapes from TFRecord:")
    print("Image shape:", image_data.shape)
    print("Track shape:", track_data.shape)
    print("Visibility shape:", visibility_data.shape)
    print("SAM shape:", sam_data.shape)
    print("Depth shape:", depth_data.shape)
    print("Instruction:", instruction)
    print("Action:", action)
    
    # Prepare results in the exact format expected by the visualization function
    results = {
        'image': image_rgb,
        'tracks': track_data,
        'visibility': visibility_data,
        'sam_features': sam_data,
        'depth_features': depth_data,
        'instruction': instruction
    }
    
    # Visualize using the original function format
    visualize_features_tfrecord(results, "tfrecord_features_visualization.png")
    
    # Also save the original image
    # image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('tfrecord_image.jpg', image_bgr)
    # print("Original image saved as: tfrecord_image.jpg")

if __name__ == "__main__":
    main()