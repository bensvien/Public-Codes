# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:46:49 2025

@author: Melly
"""
import os
# Set the environment variable to help with memory fragmentation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
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
# device='cpu'
torch.cuda.empty_cache()

#%%
cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)

print(f"CoTracker3 is using device: {device}")
#%%
# Load the video frames
video_path = 'IMG_7296.mp4'
frames = iio.imread(video_path, plugin='FFMPEG')  # Read video frames FFMPEG must be capitalised
# Convert frames to a tensor and move to the appropriate device
#%%
idx = 250
plt.imshow(frames[idx])
plt.title(f"Preview - Frame at index {idx}")
plt.axis("off")
plt.show()
#%%
frames = frames[170:240]#70 frames ok
video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # Shape: [1, T, C, H, W]

grid_size = 40  # Defines a 10x10 grid of points to track

print("Running CoTracker3...")
pred_tracks, pred_visibility = cotracker(video_tensor, grid_size=grid_size)  # Output shapes: [1, T, N, 2], [1, T, N, 1]

pred_tracks_np = pred_tracks[0].cpu().numpy()  # Shape: [T, N, 2]

#%%

# Function to visualize tracking results on a frame
def visualize_tracking(frame, tracks, frame_idx):
    """
    Visualize the tracking results on a given frame.

    Parameters:
      frame: The image (numpy array) for the current frame.
      tracks: A numpy array of shape [T, N, 2] containing the tracked coordinates.
      frame_idx: The current frame index (int) to visualize up to.
    """
    plt.imshow(frame)
    num_points = tracks.shape[1]  # number of tracked points
    for point_idx in range(num_points):
        # Plot the trajectory of each point from frame 0 to frame_idx.
        plt.plot(tracks[:frame_idx+1, point_idx, 0],
                 tracks[:frame_idx+1, point_idx, 1],
                 marker='o', markersize=3, linestyle='-', alpha=0.5)
    plt.title(f"Frame {frame_idx}")
    plt.axis("off")
    plt.show()
    
def visualize_tracking2(frame, tracks, frame_idx):
    """
    Visualize tracking results by plotting trajectories up to the specified frame.
    
    Parameters:
      frame: The current frame image (numpy array).
      tracks: A numpy array of shape [T, N, 2] with tracked coordinates.
      frame_idx: The frame index up to which to plot trajectories.
    """
    plt.imshow(frame)
    # Vectorized plotting of trajectories:
    # tracks[:frame_idx+1, :, 0] has shape [frame_idx+1, N]; transposing it to [N, frame_idx+1] 
    # makes each row a trajectory of a single point.
    plt.plot(tracks[:frame_idx+1, :, 0].T,
             tracks[:frame_idx+1, :, 1].T,
             marker='o', markersize=3, linestyle='None', alpha=0.5)
    plt.title(f"Frame {frame_idx}")
    plt.axis("off")
    plt.show()    

#%%
from cotracker.utils.visualizer import Visualizer


vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video_tensor, pred_tracks, pred_visibility)
#%%
# Visualize the first frame with tracking
frame_idx = 10  # Change to any frame index to see tracking progress
visualize_tracking2(frames[frame_idx], pred_tracks_np, frame_idx)

print("Tracking completed successfully!")
