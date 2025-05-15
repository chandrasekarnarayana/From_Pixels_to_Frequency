# From Pixels to Frequency: CNN, VGG16, and ResNet-50 Performance on Spatial, FFT, and Wavelet Domains of MNIST and Fashion MNIST

## Overview

This repository accompanies our study, "From Pixels to Frequency," which explores how different input representations—spatial, frequency (FFT), and wavelet domains—affect the performance of convolutional neural networks (CNNs) on image classification tasks.

We evaluate three CNN architectures:

* **Simple CNN**: A custom-designed network trained from scratch.
* **VGG16**: A pretrained model with frozen convolutional layers.
* **ResNet-50**: Another pretrained model with frozen convolutional layers.

These models are tested on two standard datasets: MNIST and Fashion MNIST.

## Input Representations

We preprocess the datasets into three distinct forms:

1. **Original Grayscale Images**: The raw 28×28 single-channel images.
2. **FFT Transformed Images**: Images converted to the frequency domain using the Fast Fourier Transform, capturing magnitude and phase information.
3. **Wavelet Transformed Images**: Images processed using discrete wavelet transforms to capture both spatial and frequency information.

## Key Findings

* **Wavelet Transforms Excel in Accuracy and Robustness**
Across both MNIST and Fashion MNIST, wavelet-transformed inputs consistently led to the highest classification accuracies—e.g., 99.36% for CNN and 97.07% for VGG16 on MNIST—outperforming raw spatial and FFT representations. Wavelets capture multi-scale, localized spatial-frequency features that better align with convolutional architectures.

* **FFT Representations Are Less Effective**
Models trained on FFT-transformed data performed poorly, particularly with VGG16 and ResNet-50. The global nature of FFT fails to preserve local structure, which is essential for tasks like digit and fashion item classification. This misalignment led to overfitting on certain classes (e.g., digit "1" or shirt in Fashion MNIST) and poor generalization.

* **Simple CNNs Outperform Pretrained Models in Adaptability**
The custom CNN, trained from scratch, adapted more effectively to different data representations. Its domain-specific learning outperformed transfer-learned VGG16 and ResNet-50, especially for wavelet inputs, highlighting the strength of task-tuned architectures over general-purpose pretrained ones.

* **Dimensionality Reduction Techniques Are Not Predictive of Performance**
PCA and t-SNE projections showed poor class separability across all domains (spatial, FFT, wavelet), with low silhouette scores. However, this lack of separability did not correlate with model performance—CNNs still achieved high accuracy, underscoring that deep models can learn meaningful representations even when class boundaries are ambiguous in low-dimensional space.

* **Wavelets Enhance Robustness Against Data Corruption**
Under simulated input corruption (up to 99% pixel drop), wavelet-based models (especially CNN and VGG16) retained better performance compared to FFT and even spatial inputs. This robustness makes wavelet preprocessing ideal for real-world applications like medical imaging, remote sensing, and degraded video feeds.

* **Simple Haar Transform is Sufficient**
Notably, these performance gains were achieved using a basic level-1 Haar wavelet decomposition—without complex sub-band fusion or stacking methods. This simplicity makes the approach lightweight, interpretable, and suitable for resource-constrained environments.

## Repository Structure

* `core_files/`: Contains the main scripts for data preprocessing, model training, and evaluation.
* `accuracy_data_loss_robustness/`: Includes metrics and plots related to model accuracy and loss across different input representations.
* `Confusion_matrix/`: Stores confusion matrices for each model and input type.
* `Output_metrics/`: Provides detailed performance metrics for all experiments.
* `old_experiments/`: Archives previous versions of experiments and scripts.

---

## License

This project is licensed under the MIT License.
