# Building Segmentation using U-Net (PyTorch)

Deep learning project for binary image segmentation of buildings from aerial imagery using a custom U-Net architecture built with PyTorch.

## Project Overview

This project implements an end-to-end semantic segmentation pipeline to detect buildings from aerial images using a custom U-Net model.

The model was trained on the Massachusetts Buildings Dataset and includes modern deep learning practices such as:

- Custom U-Net architecture
- Focal Loss + Dice Loss
- Mixed Precision Training (AMP)
- AdamW Optimizer
- Learning Rate Scheduling
- Early Stopping
- Data Augmentation
- IoU-based Evaluation
- Prediction Visualization

---

## Dataset

**Massachusetts Buildings Dataset**

Dataset source:
https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset

> Dataset licensing and usage rights belong to the original dataset owners.

---

## Model Architecture

Custom U-Net encoder-decoder architecture with skip connections:

- Encoder:
  - Double convolution blocks
  - Batch normalization
  - ReLU activation
  - Dropout regularization
  - Max pooling

- Bottleneck:
  - Deep feature extraction layer

- Decoder:
  - Transposed convolution upsampling
  - Skip connections
  - Feature concatenation

---

## Tech Stack

- Python
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- tqdm

---

## Training Features

### Loss Function
Hybrid loss:
- Focal Loss
- Dice Loss

Combined for better segmentation performance.

### Optimization
- AdamW optimizer
- ReduceLROnPlateau scheduler

### Performance Improvements
- Mixed precision GPU training
- Data augmentation
- Early stopping

---

## Project Structure

building-segmentation-unet/
│
├── unet.py
├── requirements.txt
├── README.md
└── sample_output.docx

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vishtodev/Building-segmentation-using-U-Net-architecture-with-PyTorch-on-Massachusetts-Buildings-Dataset.git
cd Building-segmentation-using-U-Net-architecture-with-PyTorch-on-Massachusetts-Buildings-Dataset
```
