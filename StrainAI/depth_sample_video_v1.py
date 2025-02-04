# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:47:45 2025

@author: Melly
"""


import torch
import depth_pro
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load DepthPro model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = depth_pro.create_model_and_transforms()
model.to(device)
model.eval()

# Define input/output folders
input_folder = "frames"
output_folder = "depth_frames"
os.makedirs(output_folder, exist_ok=True)

# Process each extracted frame
frame_files = sorted(os.listdir(input_folder))  # Ensure frames are processed in order

for frame_file in frame_files:
    image_path = os.path.join(input_folder, frame_file)

    # Load and preprocess the image
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)

    depth = prediction["depth"].to("cpu").squeeze().detach().numpy()

    # Save the depth map as an image
    depth_image = (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype(np.uint8)
    depth_filename = os.path.join(output_folder, f"depth_{frame_file}")
    plt.imsave(depth_filename, depth_image, cmap="jet_r")

    print(f"Processed: {frame_file} → {depth_filename}")

print(f"Depth maps saved in: {output_folder}")

