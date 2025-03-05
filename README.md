## Project Overview

This project investigates and compares the performance of three convolutional neural network (CNN) architectures on two popular image classification datasets—MNIST and Fashion MNIST. The architectures include:

- **Simple CNN:** A baseline model designed from scratch.
- **VGG16:** A pretrained model using transfer learning.
- **ResNet50:** Another powerful pretrained model using transfer learning.

The goal is to evaluate how these architectures perform across three types of input representations:
  
1. **Original Grayscale Images:** The raw 28×28 single-channel images.
2. **FFT Transformed Images:** Images transformed into the frequency domain using the Fast Fourier Transform (FFT), with features captured as magnitude and phase channels.
3. **Wavelet Transformed Images:** Images processed using discrete wavelet transforms to capture multi-scale features through approximation and detail coefficients.

---

## Key Objectives

- **Model Comparison:**  
  Benchmark and contrast a simple CNN (built from scratch) against advanced transfer learning architectures (VGG16 and ResNet50) to understand the trade-offs between model complexity and performance.

- **Input Representation Impact:**  
  Determine the effect of different image transformation techniques (original, FFT, and wavelet) on classification performance.

- **Dataset Generalization:**  
  Analyze how well each model performs on two distinct datasets (MNIST and Fashion MNIST), which vary in content complexity and visual features.

---

## Dataset

- **MNIST:**  
  - **Content:** Handwritten digits (0–9).  
  - **Properties:** Grayscale images of size 28×28, simple visual features.

- **Fashion MNIST:**  
  - **Content:** Clothing items (e.g., shirts, trousers, sneakers).  
  - **Properties:** Grayscale images of size 28×28, with more complex textures and shapes than MNIST.

*Both datasets undergo normalization (scaling pixel values to [0, 1]) and reshaping (adding an explicit channel dimension) to prepare them for model input.*

---

## Methodology

### Data Processing

- **Normalization & Reshaping:**  
  Convert pixel values to float32 in the range [0, 1] and reshape images from (28, 28) to (28, 28, 1).

- **Feature Extraction Techniques:**
  - **Original Images:**  
    Use the preprocessed grayscale images directly.
  - **FFT Transformation:**  
    Apply a 2D FFT on the single-channel images. Extract and combine magnitude and phase information into a two-channel image.
  - **Wavelet Transformation:**  
    Apply discrete wavelet transforms (e.g., using Haar wavelets) to decompose images. Resize and stack the approximation and detail coefficients to form a four-channel image.

- **Input Transformation for Transfer Learning:**  
  The models based on VGG16 and ResNet50 require input images of shape (224, 224, 3). A dedicated transformer (built with a resizing layer and a convolutional adjustment, if needed) converts the raw input images (whether 1, 2, or 4 channels) into the required format.

### Experimental Design

- **Architectures:**
  - **Simple CNN:**  
    A custom-built model that serves as a baseline for performance comparisons.
  - **VGG16 & ResNet50:**  
    Pretrained models adapted via transfer learning. Their base layers are frozen, and custom fully connected layers are added for the final classification task.

- **Training Regimen:**
  - **Datasets:** Experiments are run on both MNIST and Fashion MNIST.
  - **Data Representations:** For each dataset, separate models are trained on the original images, FFT-transformed images, and wavelet-transformed images.
  - **Callbacks:**  
    - *EarlyStopping* to avoid overfitting.
    - *ModelCheckpoint* to store the best-performing model.
    - *ReduceLROnPlateau* to adjust the learning rate dynamically.

- **Evaluation Metrics:**
  - **Accuracy:** Overall percentage of correctly classified samples.
  - **Classification Report:** Includes precision, recall, and F1-score for each class.
  - **Confusion Matrix:** Detailed breakdown of prediction errors to identify misclassification patterns.

### Metrics of Evaluation

Each trained model is evaluated using:
- **Overall Test Accuracy:** How well the model generalizes on unseen data.
- **Classification Report & Confusion Matrix:** Provide insights into class-specific performance and misclassifications.

---

## Deliverables

- **Model Checkpoints:**  
  Saved models (in `.keras` format) for each architecture (Simple CNN, VGG16, ResNet50) and each data representation (original, FFT, wavelet) on both MNIST and Fashion MNIST.

- **Evaluation Outputs:**  
  Detailed classification reports and confusion matrices for each experimental branch.

- **Jupyter Notebook Documentation:**  
  A well-commented notebook that details all aspects of data processing, model building, training, and evaluation. This document serves as both a record of the experiments and a reproducible resource for future work.

- **Comparative Analysis Report:**  
  A summary comparing the performance of the three architectures across different datasets and data representations, including insights on which combinations yield the best results and recommendations for future research.

---

## License

This project is licensed under the MIT License.
