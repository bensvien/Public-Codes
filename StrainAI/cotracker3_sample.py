# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:46:49 2025

@author: Melly
"""
# pip install torch imageio[ffmpeg]
import torch
import numpy as np

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)

print(f"CoTracker3 is using device: {device}")
#%%

import imageio.v3 as iio
# Load the video frames
video_path = 'IMG_7295.mp4'
frames = iio.imread(video_path, plugin='ffmpeg')  # Read video frames
# Convert frames to a tensor and move to the appropriate device
video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # Shape: [1, T, C, H, W]

grid_size = 10  # Defines a 10x10 grid of points to track

print("Running CoTracker3...")
pred_tracks, pred_visibility = cotracker(video_tensor, grid_size=grid_size)  # Output shapes: [1, T, N, 2], [1, T, N, 1]

pred_tracks_np = pred_tracks[0].cpu().numpy()  # Shape: [T, N, 2]

import matplotlib.pyplot as plt


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

