import tensorflow_datasets as tfds
import cv2
import random
import numpy as np

data_dir = "/data3/embodied/modified_libero_rlds_feats/tensorflow_datasets/libero_spatial_no_noops/1.0.0"

ds = tfds.builder_from_directory(data_dir).as_dataset(split='train')

episode = next(iter(ds.take(1)))

steps = list(tfds.as_numpy(episode['steps']))

random_step = random.choice(steps)

image_data = random_step['observation']['image']

track_data = random_step['observation']['tracks_image']
sam_data = random_step['observation']['sam_features_image']
depth_data = random_step['observation']['depth_features_image']
instruction = random_step['language_instruction']

image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('random_image_tf.jpg', image_bgr)

print("Image shape:", image_data.shape)
print("Track shape:", track_data.shape)
print("SAM shape:", sam_data.shape)
print("Depth shape:", depth_data.shape)
print("Instruction:", instruction)