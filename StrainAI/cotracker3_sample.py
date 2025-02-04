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
import time

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
video_path = 'IMG_7296.mp4'
frames = iio.imread(video_path, plugin='FFMPEG')  # Read video frames FFMPEG must be capitalised

#%% Check Frame Image
idx = 250
plt.imshow(frames[idx])
plt.title(f"Preview - Frame at index {idx}")
plt.axis("off")
plt.show()
#%%# Convert frames to a tensor and move to the appropriate device
frames = frames[1:480:64]#Default for GPU 70 Frames/40 Gridsize
video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # Shape: [1, T, C, H, W]
grid_size = 40  # Defines a NxN grid of points to track

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
tic = time.time()  # Start timing
pred_tracks, pred_visibility = cotracker(video_tensor, grid_size=grid_size)  # Output shapes: [1, T, N, 2], [1, T, N, 1]
# pred_tracks, pred_visibility = cotracker(video_tensor, queries=custom_queries)  # Output shapes: [1, T, N, 2], [1, T, N, 1]
toc = time.time()  # End timing
elapsed_time = toc - tic
print(f"Elapsed time: {elapsed_time:.6f} seconds")
#%%
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
    
import matplotlib.cm as cm  # Import colormap utilities

def plot_single_frame_with_gradient(frame, tracks, frame_idx):
    """
    Plots a single video frame with overlaid tracking points in gradient colors.

    Parameters:
    - frame: The image (NumPy array) of the frame to be plotted.
    - tracks: A NumPy array containing the tracked points' coordinates.
    - frame_idx: The index of the frame to visualize.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)

    num_points = tracks.shape[1]  # Number of tracked points
    colors = cm.jet_r(np.linspace(0, 1, num_points))  # Generate a smooth gradient colormap

    for point_idx in range(num_points):
        x, y = tracks[frame_idx, point_idx]  # Get (x, y) coordinates
        plt.scatter(x, y, color=colors[point_idx], s=20, edgecolors='None', linewidth=0.5)  # Gradient color dots

    plt.title(f"Tracking Visualization - Frame {frame_idx}")
    plt.axis("off")
    plt.show()
    
def plot_single_frame_with_gradient_visibility(frame, tracks, visibility, frame_idx):
    """
    Plots a single video frame with tracking points, using gradient colors and visibility weighting.

    Parameters:
    - frame: The image (NumPy array) of the frame to be plotted.
    - tracks: A NumPy array containing the tracked points' coordinates.
    - visibility: A NumPy array containing visibility scores (confidence).
    - frame_idx: Index of the frame to visualize.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)

    num_points = tracks.shape[1]  # Number of tracked points
    colors = cm.jet_r(np.linspace(0, 1, num_points))  # Generate gradient colormap

    # Remove batch dimension & extract visibility scores
    visibility_np = visibility.squeeze(0).cpu().numpy()  # Shape becomes [T, N]
    visibilities = visibility_np[frame_idx, :].astype(np.float32)  # Convert to float

    # Normalize visibility values between 0 and 1
    visibilities = (visibilities - visibilities.min()) / (visibilities.max() - visibilities.min() + 1e-6)

    for point_idx in range(num_points):
        x, y = tracks[frame_idx, point_idx]  # Get (x, y) coordinates
        alpha = visibilities[point_idx]  # Set transparency based on visibility
        size = 5 + 20 * alpha  # Scale dot size (more visible points are larger)

        plt.scatter(x, y, color=colors[point_idx], s=size, alpha=alpha, edgecolors='None', linewidth=0.5)

    plt.title(f"Tracking Visualization - Frame {frame_idx} (Visibility Weighted)")
    plt.axis("off")
    plt.show()

# Example usage:
frame_index = 5  # Index of the frame you want to plot
plot_single_frame_with_gradient_visibility(frames[frame_index], pred_tracks_np, pred_visibility, frame_index)
#%%
# Example usage:
frame_index = 5  # Index of the frame you want to plot
plot_single_frame_with_gradient(frames[frame_index], pred_tracks_np, frame_index)
    
#%% Call cotracker Visualizer
from cotracker.utils.visualizer import Visualizer
vis = Visualizer(save_dir="./saved_videosTEST", pad_value=1, linewidth=3)
vis.visualize(video_tensor, pred_tracks, pred_visibility)
#%%
# Visualize the first frame with tracking
# Set a valid save directory
frame_idx = 0  # First frame
visualize_tracking2(frames[frame_idx], pred_tracks_np, frame_idx)
# vis2 = Visualizer(save_dir=save_dir, pad_value=1, linewidth=3)  # save_dir=None prevents file saving


print("Tracking completed successfully!")
