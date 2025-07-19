# Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning
Diagnose 14 pathologies on Chest X-Ray using Deep Learning. Perform diagnostic interpretation using GradCAM Method



# Project Title: 
X-ray Lung Classifier – Deep Learning-Based Medical Imaging Application
Tech Stack: Python, PyTorch, Streamlit, Torchvision, NumPy, PIL, Matplotlib
Deployment: Render / Hugging Face / Local (optional)

# Model Architecture Summary – Custom CNN for X-ray Lung Classification
Model Name: Net
Framework: PyTorch
Purpose: Classify chest X-ray images into two categories (e.g., Normal vs Pneumonia)
Input Shape: 3-channel (RGB) chest X-ray images
Output: Log-softmax scores for 2 classes


# Architecture Overview

This CNN is a deep, custom-built architecture that progressively extracts spatial features using a combination of convolutional, pooling, and normalization layers. Key aspects include:

Layer Block	Description
convolution_block1	Conv2d (3→8), ReLU, BatchNorm
pooling11	MaxPool2d (2×2)
convolution_block2	Conv2d (8→20), ReLU, BatchNorm
pooling22	MaxPool2d (2×2)
convolution_block3	Conv2d (20→10, 1×1), ReLU, BatchNorm
pooling33	MaxPool2d (2×2)
convolution_block4	Conv2d (10→20), ReLU, BatchNorm
convolution_block5	Conv2d (20→32, 1×1), ReLU, BatchNorm
convolution_block6	Conv2d (32→10), ReLU, BatchNorm
convolution_block7	Conv2d (10→10, 1×1), ReLU, BatchNorm
convolution_block8	Conv2d (10→14), ReLU, BatchNorm
convolution_block9	Conv2d (14→16), ReLU, BatchNorm
gap	Global Average Pooling (4×4)
convolution_block_out	Conv2d (16→2, 4×4)
output	Flatten to shape (-1, 2), Log-Softmax

