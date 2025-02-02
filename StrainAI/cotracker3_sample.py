# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:46:49 2025

@author: Melly
"""
import os
# Set the environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# pip install torch imageio[ffmpeg]
import torch
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.empty_cache()

#%%
cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)

print(f"CoTracker3 is using device: {device}")
#%%
# Load the video frames
video_path = 'IMG_7295.mp4'
frames = iio.imread(video_path, plugin='FFMPEG')  # Read video frames FFMPEG must be capitalised
# Convert frames to a tensor and move to the appropriate device
video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # Shape: [1, T, C, H, W]

grid_size = 4  # Defines a 10x10 grid of points to track

print("Running CoTracker3...")
pred_tracks, pred_visibility = cotracker(video_tensor, grid_size=grid_size)  # Output shapes: [1, T, N, 2], [1, T, N, 1]

pred_tracks_np = pred_tracks[0].cpu().numpy()  # Shape: [T, N, 2]


# Function to visualize tracking results on a frame
def visualize_tracking(frame, tracks, frame_idx):
    plt.imshow(frame)
    for track in tracks:
        plt.plot(track[:frame_idx+1, 0], track[:frame_idx+1, 1], marker='o', markersize=3, linestyle='-', alpha=0.5)
    plt.title(f"Frame {frame_idx}")
    plt.axis("off")
    plt.show()

# Visualize the first frame with tracking
frame_idx = 0  # Change to any frame index to see tracking progress
visualize_tracking(frames[frame_idx], pred_tracks_np, frame_idx)

print("Tracking completed successfully!")
