# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:35:38 2025

@author: Melly
"""

# git clone https://github.com/apple/ml-depth-pro.git
# cd ml-depth-pro
# pip install .
import time
from PIL import Image
import depth_pro
import matplotlib.pyplot as plt

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()
#%
image_path = "./input_image.jpg"
# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

tic = time.time()  # Start timing
# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

toc = time.time()  # End timing
elapsed_time = toc - tic
print(f"Elapsed time: {elapsed_time:.6f} seconds")
#%%
# Convert depth tensor to numpy array for visualization
# depth_np = depth.squeeze().detach().cpu().numpy() if depth.is_cuda else depth.squeeze().detach().numpy()
depth_np =depth.squeeze().detach().numpy()
# Plot the depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_np, cmap="plasma")
plt.colorbar(label="Depth (meters)")
plt.title("Estimated Depth Map")
plt.axis("off")  # Hide axes for better visualization
plt.show()
