# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:35:38 2025

@author: Melly
"""

# git clone https://github.com/apple/ml-depth-pro.git
# cd ml-depth-pro
# pip install .
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from PIL import Image
import depth_pro
import matplotlib.pyplot as plt
import torch
print("CUDA Enabled:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("Supports FP16:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import matplotlib
#matplotlib.use('Agg')  # Avoid GUI rendering overhead

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.to(device)
model.eval()
#%
if torch.__version__ >= "2.0":
    model = torch.compile(model)  # Enables graph-level optimization
image_path = "./input_image.jpg"
# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
# image = transform(image)
image = transform(image).to(device)

if device.type == "cuda": #Floating Points 16-bit pecision
    image = image.half()
    model = model.half()

tic = time.time()  # Start timing
# Run inference.
# prediction = model.infer(image, f_px=f_px)
with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
depth = prediction["depth"].to("cpu")

toc = time.time()  # End timing
elapsed_time = toc - tic
print(f"Elapsed time: {elapsed_time:.6f} seconds")
#%%
# Convert depth tensor to numpy array for visualization
depth_np = depth.squeeze().detach().cpu().numpy() if depth.is_cuda else depth.squeeze().detach().numpy()

# matplotlib.use('TkAgg') 
# Plot the depth map with a fixed color scale 
plt.figure(figsize=(8, 6))
plt.imshow(depth_np, cmap="jet_r", vmin=2.3, vmax=3.7)  # Set scale range
plt.colorbar(label="Depth (metres)")
plt.title("Estimated Depth Map")
plt.axis("off")  # Hide axes for better visualization
# Save the figure
plt.savefig("depth_map.png", dpi=800, bbox_inches="tight")  # High-quality PNG


