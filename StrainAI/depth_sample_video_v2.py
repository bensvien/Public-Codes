# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:38:39 2025

@author: Melly
"""
"""
Optimized Depth Estimation from Video using DepthPro
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import depth_pro
import os
import cv2
import numpy as np
from PIL import Image

# Load DepthPro model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Enabled:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("Supports FP16:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")
torch.cuda.empty_cache()

model, transform = depth_pro.create_model_and_transforms()
model.to(device)
model.eval()

if torch.__version__ >= "2.0":
    model = torch.compile(model)  # Enables graph-level optimization

#%%
# Define input folder and output video path
input_folder = "frames"
output_video_path = "depth_output5.mp4"

# Get frame list
frame_files = sorted(os.listdir(input_folder))  # Ensure frames are processed in order

# Read first frame to get dimensions
first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
height, width, _ = first_frame.shape

# Define video writer (Single save at the end)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30  # Adjust as needed
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)


#%%
# Store depth maps as GPU tensors
depth_frames = []
global_min, global_max = float('inf'), float('-inf')  # Track min/max across all frames

# Process frames & store on GPU
for frame_file in frame_files:
    image_path = os.path.join(input_folder, frame_file)

    # Load and preprocess the image
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image).to(device)
    
    if device.type == "cuda": #Floating Points 16-bit pecision (This helps a lot)
        image = image.half()
        model = model.half()

#if CRASH
    # with torch.cuda.amp.autocast():
    #   prediction = model.infer(image, f_px=f_px)

    # Run inference
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)

    depth = prediction["depth"].squeeze()  # Keep on GPU

    # Update global min/max (done in GPU)
    frame_min = torch.min(depth)
    frame_max = torch.max(depth)
    global_min = min(global_min, frame_min.item())
    global_max = max(global_max, frame_max.item())

    # Append to depth frame list (still on GPU)
    depth_frames.append(depth)

    print(f"Processed: {frame_file}")

#%%
import matplotlib.pyplot as plt

# Set your custom (X, Y) reference point
baseline_x, baseline_y = 200, 1800  # Change this to any point

# Convert depth maps to color and save video
for i, depth_frame in enumerate(depth_frames):
    depth_np = depth_frame.cpu().numpy()  # Move from GPU to CPU
    depth_np = np.squeeze(depth_np)  # Ensure shape is (H, W)

    # Ensure (X, Y) is within bounds
    if 0 <= baseline_y < depth_np.shape[0] and 0 <= baseline_x < depth_np.shape[1]:
        baseline_depth = depth_np[baseline_y, baseline_x]  # Extract depth at (X, Y)
    else:
        print(f"Warning: (X={baseline_x}, Y={baseline_y}) is out of bounds! Using median depth instead.")
        baseline_depth = np.median(depth_np)  # Fallback to median if out of bounds

    print(f"Frame {i} - Baseline Depth at ({baseline_x},{baseline_y}): {baseline_depth:.4f}")

    # Scale depth map using baseline depth (Preserve original depth relations)
    depth_np = (depth_np / baseline_depth) * 255
    depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)  # Convert to uint8

    # Apply OpenCV Jet colormap (BGR format)
    depth_colored = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
    depth_colored = cv2.bitwise_not(depth_colored)
    
    depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    # Display a sample frame every 100 frames with dynamic reference point
    if i % 100 == 0:
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_colored_rgb)
        plt.scatter(baseline_x, baseline_y, color="red", s=100, edgecolors="black", marker="o", label=f"Baseline ({baseline_x},{baseline_y})")  #  Mark reference point
        plt.axis("off")
        plt.title(f"Frame {i} - Depth Map Scaled by Baseline")
        plt.legend()
        plt.show()

    # Resize frame if needed
    if depth_colored.shape[:2] != (height, width):
        depth_colored = cv2.resize(depth_colored, (width, height))

    # Write corrected frame to video
    out.write(depth_colored)
    print(f"Added frame {i} to video")

# Release video writer
out.release()
print(f"Depth video saved successfully as: {output_video_path}")




#%% Simple
# Normalize depth maps using overall min/max (Done in GPU)
depth_frames_normalized = [(255 * (d - global_min) / (global_max - global_min)).to(torch.uint8) for d in depth_frames]
import matplotlib.pyplot as plt
#% Debug
for i, depth_frame in enumerate(depth_frames_normalized):
    depth_np = depth_frame.cpu().numpy()  # Move from GPU to CPU


    depth_np = np.squeeze(depth_np)
    # depth_np = np.clip(depth_np / depth_np.max() * 255, 0, 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
    # depth_colored = cv2.applyColorMap(depth_np.astype(np.uint8), cv2.COLORMAP_JET)

    if i % 100==0:
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_colored,cmap="jet", vmin=0, vmax=255)
        plt.axis("off")
        plt.title(f"Frame {i} - Corrected Depth Map")
        plt.show()

    #  Resize frame if needed
    if depth_colored.shape[:2] != (height, width):
        depth_colored = cv2.resize(depth_colored, (width, height))

    #  Write corrected frame to video
    out.write(depth_colored)
    print(f"Added frame {i} to video")

#  Release video writer
out.release()
print(f"Depth video saved successfully as: {output_video_path}")



#%%
# #%% Old Version
# # Convert frames to color using `jet_r` and save video
# for depth_frame in depth_frames_normalized:
#     out.write(depth_frame.cpu().numpy())  # Convert from GPU to CPU only once

# # Release video writer
# out.release()
# print(f"Depth video saved successfully as: {output_video_path}")

