import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import streamlit as st
from PIL import Image

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("acne_model.pth"))
model.eval() 

print(model)
