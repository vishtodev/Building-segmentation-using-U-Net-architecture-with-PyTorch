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

```bash
building-segmentation-unet/
│
├── unet.py
├── requirements.txt
├── README.md
└── sample_results/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/building-segmentation-unet.git
cd building-segmentation-unet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Update dataset paths in:

```python
BASE_PATH = "./dataset/png"
```

Then run:

```bash
python unet.py
```

---

## Results

Evaluation metric used:
**Intersection over Union (IoU)**

Example outputs:

(Add screenshots of predictions here)

---

## Future Improvements

- Use pretrained encoder (ResNet / EfficientNet)
- Add Dice coefficient tracking
- Improve hyperparameter tuning
- Deploy as web demo
- Experiment with Attention U-Net

---

## Notes

- Kaggle API credentials have been removed for security.
- Parts of this implementation were developed with AI assistance and customized for this project.

---

## License

This project is licensed under the MIT License.

---

## Author

**Vishal R**

Electronics and Communication Engineering Student  
Interested in AI, Computer Vision, and Embedded Systems
