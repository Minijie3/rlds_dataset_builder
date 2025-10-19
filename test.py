import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Import feature extraction functions
from LIBERO_Goal.LIBERO_Goal_dataset_builder import extract_tracks_for_episode, extract_sam_features, extract_depth_features, extract_laq_features

def test_feature_extraction(image_path):
    """Test feature extraction for a single image"""
    
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Duplicate the image to simulate multiple frames (track extraction requires multiple frames)
    num_frames = 10  # Simulate 10 frames
    images = [image] * num_frames
    
    print(f"Test image size: {image.shape}")
    
    # Test track extraction
    print("Extracting track features...")
    tracks, visibility = extract_tracks_for_episode(images, frame_gap=8, patch_size=8)
    print(f"Track feature shape: {tracks.shape}")
    print(f"Visibility feature shape: {visibility.shape}")
    
    # Test SAM features
    print("Extracting SAM features...")
    sam_features, masks_list = extract_sam_features([image])  # Single image
    print(f"SAM feature shape: {sam_features.shape}")
    print(f"Number of masks generated: {len(masks_list[0])}")
    
    # Test depth features
    print("Extracting depth features...")
    depth_features = extract_depth_features([image])  # Single image
    print(f"Depth feature shape: {depth_features.shape}")
    
    # Test LAQ features for different modalities
    print("\nExtracting LAQ features...")
    
    # LAQ for image features
    print("Extracting LAQ image features...")
    laq_image_features = extract_laq_features(images, feature_type='image')
    print(f"LAQ image features shape: {laq_image_features.shape}")
    
    # LAQ for depth features
    print("Extracting LAQ depth features...")
    # Note: We need to create depth features for all frames first
    depth_features_all = extract_depth_features(images)
    laq_depth_features = extract_laq_features(depth_features_all, feature_type='depth')
    print(f"LAQ depth features shape: {laq_depth_features.shape}")
    
    # LAQ for SAM features
    print("Extracting LAQ SAM features...")
    # Note: We need to create SAM features for all frames first
    sam_features_all, _ = extract_sam_features(images)
    laq_sam_features = extract_laq_features(sam_features_all, feature_type='sam')
    print(f"LAQ SAM features shape: {laq_sam_features.shape}")
    
    return {
        'image': image,
        'tracks': tracks,
        'visibility': visibility,
        'sam_features': sam_features,
        'masks': masks_list,
        'depth_features': depth_features,
        'laq_image_features': laq_image_features,
        'laq_depth_features': laq_depth_features,
        'laq_sam_features': laq_sam_features
    }

def visualize_features(results, save_path=None):
    """Visualize the extracted features"""
    
    image = results['image'][::-1, ::-1]
    tracks = results['tracks']
    visibility = results['visibility']
    sam_features = results['sam_features']
    masks = results['masks']
    depth_features = results['depth_features']
    
    # LAQ features for printing shapes
    laq_image_features = results['laq_image_features']
    laq_depth_features = results['laq_depth_features']
    laq_sam_features = results['laq_sam_features']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Track visualization (show tracks in the last frame)
    last_frame_tracks = tracks[0]  # Tracks in the 1st frame
    axes[0, 1].imshow(Image.fromarray(image).resize((224, 224), resample=Image.BICUBIC))
    # Draw track points on the image
    for i, (dx, dy) in enumerate(last_frame_tracks):
        if i % 5 == 0:  # Draw one point every 20 points to avoid overcrowding
            x = (i % 28) * 8 + 4  # Calculate the original position (center of the grid)
            y = (i // 28) * 8 + 4
            axes[0, 1].arrow(x, y, dx, dy, head_width=3, head_length=2, fc='red', ec='red')
    axes[0, 1].set_title('Optical Flow Tracks')
    axes[0, 1].axis('off')
    
    # Visibility heatmap
    last_frame_visibility = visibility[-1].reshape(28, 28)
    im = axes[0, 2].imshow(last_frame_visibility, cmap='hot')
    axes[0, 2].set_title('Visibility Heatmap')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # SAM feature visualization (take the average of the first channel)
    sam_avg = sam_features[0].mean(axis=0).reshape(16, 16)
    im = axes[1, 0].imshow(sam_avg, cmap='viridis')
    axes[1, 0].set_title('SAM Features (avg)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # SAM Mask visualization
    axes[1, 1].imshow(image)
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
    for i, mask_data in enumerate(masks):
        segmentation = mask_data['segmentation']
        combined_mask = np.logical_or(combined_mask, segmentation)
        from skimage import measure
        contours = measure.find_contours(segmentation, 0.5)
        for contour in contours:
            axes[1, 1].plot(contour[:, 1], contour[:, 0], linewidth=2, 
                            color=colors[i], label=f'Mask {i+1}' if i < 5 else "")
    axes[1, 1].imshow(combined_mask, alpha=0.5, cmap='Reds')
    if len(masks) <= 5:
        axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].set_title(f'SAM Masks ({len(masks)} masks)')
    axes[1, 1].axis('off')
    
    # Depth feature visualization
    if len(depth_features[0].shape) == 2:
        axes[1, 2].imshow(depth_features[0], cmap='gray')
    else:
        depth_avg = depth_features[0].mean(axis=0)
        axes[1, 2].imshow(depth_avg, cmap='gray')
    axes[1, 2].set_title('Depth Features')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    masks_shape_info = "N/A"
    first_mask_shape = masks[0]['segmentation'].shape
    masks_shape_info = f"{len(masks)} masks, each {first_mask_shape}"
    
    stats_text = f"""
    Feature Statistics:
    - Tracks shape: {tracks.shape}
    - Visibility shape: {visibility.shape}
    - SAM features shape: {sam_features.shape}
    - Masks: {masks_shape_info}
    - Depth features shape: {depth_features.shape}
    - LAQ image features shape: {laq_image_features.shape}
    - LAQ depth features shape: {laq_depth_features.shape}
    - LAQ SAM features shape: {laq_sam_features.shape}
    
    Value Ranges:
    - Tracks: [{tracks.min():.3f}, {tracks.max():.3f}]
    - Visibility: [{visibility.min():.3f}, {visibility.max():.3f}]
    - SAM: [{sam_features.min():.3f}, {sam_features.max():.3f}]
    - Depth: [{depth_features.min():.3f}, {depth_features.max():.3f}]
    - LAQ Image: [{laq_image_features.min():.3f}, {laq_image_features.max():.3f}]
    - LAQ Depth: [{laq_depth_features.min():.3f}, {laq_depth_features.max():.3f}]
    - LAQ SAM: [{laq_sam_features.min():.3f}, {laq_sam_features.max():.3f}]
    """
    
    plt.figtext(0.1, 0.02, stats_text, fontsize=10, fontfamily='monospace')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization results saved to: {save_path}")
    
    plt.show()

def test_single_image(image_path, output_path="feature_visualization.png"):
    """Complete test process"""
    
    print("Starting feature extraction test...")
    print("=" * 50)
    
    try:
        # Extract features
        results = test_feature_extraction(image_path)
        
        print("\nFeature extraction completed!")
        print("=" * 50)
        
        # Print detailed LAQ feature information
        print("\nLAQ Feature Details:")
        print("-" * 30)
        print(f"LAQ Image Features: shape={results['laq_image_features'].shape}, "
              f"range=[{results['laq_image_features'].min():.3f}, {results['laq_image_features'].max():.3f}]")
        print(f"LAQ Depth Features: shape={results['laq_depth_features'].shape}, "
              f"range=[{results['laq_depth_features'].min():.3f}, {results['laq_depth_features'].max():.3f}]")
        print(f"LAQ SAM Features: shape={results['laq_sam_features'].shape}, "
              f"range=[{results['laq_sam_features'].min():.3f}, {results['laq_sam_features'].max():.3f}]")
        
        # Visualize the results (excluding LAQ features from visualization as requested)
        visualize_features(results, output_path)
        
        return results
        
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test image path - Replace with your image path
    test_image_path = "random_image.png"  # Or use another image path
    
    import os
    # If there is no test image, create a simple test image
    if not os.path.exists(test_image_path):
        print("Creating a test image...")
        # Create a simple color test image
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        print(f"Test image created: {test_image_path}")
    
    # Run the test
    results = test_single_image(test_image_path, "test_results.png")
    
    if results is not None:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")