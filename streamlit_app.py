import streamlit as st
import os
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import os

model_path = "https://github.com/PrasadCoding/AI_Skin_Score/raw/refs/heads/master/acne_model.pth"  # Update this path if the file is in a different location
if os.path.exists(model_path):
    print(f"Model file found: {model_path}")
else:
    print(f"Model file not found: {model_path}")

