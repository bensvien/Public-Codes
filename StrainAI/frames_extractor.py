# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:12:26 2025

@author: Melly
"""

import cv2
import os

# Load video
video_path = "IMG_7296.mp4"
output_folder = "frames"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Extracting {total_frames} frames...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends

    # Save frame as JPG
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1
    print(frame_count)
cap.release()
print(f"Frames saved in: {output_folder}")
