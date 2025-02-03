# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:46:49 2025

@author: Melly
"""

# !pip install torch torchvision
# !pip install imageio
# !pip install matplotlib
# !pip install imageio[ffmpeg]

import os

# Set the environment variable 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Library
import torch
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
# 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'
torch.cuda.empty_cache()

#%% Call CoTracker3 Offline Mode
cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)
print(f"CoTracker3 is using device: {device}")
#%% Load Video Frames
# Load the video frames
video_path = 'IMG_7295.mp4'
frames = iio.imread(video_path, plugin='FFMPEG')  # Read video frames FFMPEG must be capitalised

#%% Check Frame Image
idx = 250
plt.imshow(frames[idx])
plt.title(f"Preview - Frame at index {idx}")
plt.axis("off")
plt.show()
#%%# Convert frames to a tensor and move to the appropriate device
frames = frames[100:480:32]#Default for GPU 70 Frames/40 Gridsize
video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # Shape: [1, T, C, H, W]
grid_size = 80  # Defines a NxN grid of points to track

# custom_queries = torch.tensor([[
#     [0, 500, 550],  # For example, point 1: frame 0, x=100, y=150
#     [0, 600, 650],  # For example, point 2: frame 0, x=200, y=250
#     [0, 800, 850],  # For example, point 3: frame 0, x=300, y=350
#     [0, 800, 650],   # For example, point 3: frame 0, x=300, y=350
#     [0, 800, 250],   # For example, point 3: frame 0, x=300, y=350
#     [0, 500, 550],   # For example, point 3: frame 0, x=300, y=350
#     [0, 600, 1050],   # For example, point 3: frame 0, x=300, y=350
#     [0, 300, 1250],   # For example, point 3: frame 0, x=300, y=350
#     [0, 900, 1450],   # For example, point 3: frame 0, x=300, y=350
# ]], dtype=torch.float32).to(device)

print("Running CoTracker3...")
pred_tracks, pred_visibility = cotracker(video_tensor, grid_size=grid_size)  # Output shapes: [1, T, N, 2], [1, T, N, 1]
# pred_tracks, pred_visibility = cotracker(video_tensor, queries=custom_queries)  # Output shapes: [1, T, N, 2], [1, T, N, 1]
pred_tracks_np = pred_tracks[0].cpu().numpy()  # Shape: [T, N, 2]

#%% Function visualize trackering2 
def visualize_tracking2(frame, tracks, frame_idx):
    """
    Visualize tracking results by plotting trajectories up to the specified frame.
    
    Parameters:
      frame: The current frame image (numpy array).
      tracks: A numpy array of shape [T, N, 2] with tracked coordinates.
      frame_idx: The frame index up to which to plot trajectories.
    """
    plt.imshow(frame)
    plt.plot(tracks[:frame_idx+1, :, 0].T,
             tracks[:frame_idx+1, :, 1].T,
             marker='o', markersize=3, linestyle='None', alpha=0.5)
    plt.title(f"Frame {frame_idx}")
    plt.axis("off")
    plt.show()    

#%% Call cotracker Visualizer
from cotracker.utils.visualizer import Visualizer
vis = Visualizer(save_dir="./saved_videosTEST", pad_value=1, linewidth=3)
vis.visualize(video_tensor, pred_tracks, pred_visibility)
#%%
# Visualize the first frame with tracking
frame_idx = 1  # Change to any frame index to see tracking progress
visualize_tracking2(frames[frame_idx], pred_tracks_np, frame_idx)

print("Tracking completed successfully!")
