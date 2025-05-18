# BrainScope: MRI-Based Binary Classification of Tumor Presence Using Deep CNN Architectures

This project builds a complete end-to-end deep learning pipeline for classifying brain MRI scans into **Tumor** and **No Tumor** categories. It leverages well-established CNN models — **VGG16**, **InceptionV3**, and **ResNet50** — and applies customized preprocessing and evaluation strategies tailored for medical image classification.

---

## Overview

- **Input**: MRI scans from a binary-labeled dataset (`yes` / `no`)
- **Objective**: Predict tumor presence from image features
- **Models Used**: VGG16, InceptionV3, ResNet50
- **Image Size**: 224×224 (standardized)
- **Output**: Binary classification (0 = No Tumor, 1 = Tumor)

---

## Pipeline

1. **Data Splitting**  
   Images were divided into three sets:
   - `TRAIN/YES`, `TRAIN/NO`
   - `VAL/YES`, `VAL/NO`
   - `TEST/YES`, `TEST/NO`

2. **Image Preprocessing**  
   - Converted images to grayscale and denoised using Gaussian Blur
   - Extracted brain region via contour detection
   - Cropped around the extreme points of the largest contour
   - Resized cropped images to 224×224

3. **Augmentation**  
   Applied transformations on the training set including:
   - Rotation, brightness shifts, flips, shear, and zoom
   - Augmented previews generated and verified visually

4. **Model Construction**  
   - Loaded VGG16, InceptionV3, and ResNet50 (excluding their top layers)
   - Added dropout, flattening, and a final dense sigmoid layer
   - Compiled with binary crossentropy and Adam optimizer

5. **Training**  
   - Trained for 20 epochs with batch generators
   - Used separate generators for training and validation with real-time augmentation

6. **Evaluation**  
   - Computed validation accuracy after training
   - Calculated performance metrics including:
     - Accuracy, Precision, Recall, F1 Score
     - Cohen’s Kappa, ROC AUC
     - Confusion Matrix

7. **Visualization**  
   - Plotted training loss and accuracy curves for each model
   - Visualized feature maps of early convolutional layers
   - Showed confusion matrices per model on the validation set

8. **Model Saving**  
   Final models saved:
   - `2019-8-6_VGG_model.h5`
   - `2019-8-6_inception_v3.h5`
   - `2019-8-6_resnet50.h5`

---

## Output Files

- `TRAIN_CROP`, `VAL_CROP`, and `TEST_CROP`: Processed image folders
- `*_loss_plot.png`, `*_accuracy_plot.png`: Model-wise performance curves
- `.h5` model files: Trained model weights for reuse

---

## Libraries Used

- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy, Matplotlib
- scikit-learn
- imutils
