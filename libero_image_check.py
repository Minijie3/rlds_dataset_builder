import h5py
import numpy as np
import random
import cv2

file_path = "/data3/embodied/libero_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"
with h5py.File(file_path, 'r') as f:
    agentview_rgb = f['data/demo_9/obs/agentview_rgb']
    wrist_rgb = f['data/demo_9/obs/eye_in_hand_rgb']
    
    random_index = random.randint(0, len(agentview_rgb) - 1)
    
    image_data = agentview_rgb[random_index]
    wrist_data = wrist_rgb[random_index]
    
    cv2.imwrite('random_image.png', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
    cv2.imwrite('random_wrist_image.png', cv2.cvtColor(wrist_data, cv2.COLOR_RGB2BGR))