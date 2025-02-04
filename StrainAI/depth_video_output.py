# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:42:23 2025

@author: Melly
"""

import cv2
import os

input_folder = "depth_frames"
output_video = "depth_output.mp4"

# Get all depth image files
frame_files = sorted(os.listdir(input_folder))
frame = cv2.imread(os.path.join(input_folder, frame_files[0]))  # Read first frame
height, width, layers = frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30  # Adjust frame rate as needed
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each depth frame into video
for frame_file in frame_files:
    frame_path = os.path.join(input_folder, frame_file)
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()
print(f"Depth video saved as: {output_video}")
