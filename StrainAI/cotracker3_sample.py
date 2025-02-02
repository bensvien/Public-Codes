"""
Created on Sun Feb  2 20:46:49 2025

@author: Melly
"""
# pip install torch imageio[ffmpeg]
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
