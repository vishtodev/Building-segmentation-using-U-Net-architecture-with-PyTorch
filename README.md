# Building-segmentation-using-U-Net-architecture-with-PyTorch

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
## Running the Project (Google Colab)

This project is designed to run in **Google Colab with GPU acceleration**.

### Steps

1. Open Google Colab
2. Upload `Building_segmentation_using_U-Net_architecture_with_PyTorch_on_Massachusetts_Buildings_Dataset.ipynb`
3. Enable GPU:
   Runtime → Change runtime type → GPU
4. Install dependencies:
```python
!pip install albumentations opencv-python matplotlib tqdm
```

5. Configure Kaggle API credentials:
```python
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_USERNAME","key":"YOUR_KAGGLE_KEY"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

6. Download dataset:
```python
!kaggle datasets download -d balraj98/massachusetts-buildings-dataset
!unzip massachusetts-buildings-dataset.zip
```

7. Run the training script.
