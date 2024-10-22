# Evaluating FFT Input in Shallow and Pre-trained CNNs Using Keras

## Project Overview

This project aims to explore the effectiveness of using Fast Fourier Transform (FFT)-transformed inputs in shallow convolutional neural networks (CNNs) as compared to traditional spatial domain inputs. Additionally, it evaluates the benefits of combining FFT and spatial inputs and compares shallow models with deeper pre-trained models, such as ResNet. The project analyzes performance, training time, convergence, and failure cases of FFT-based inputs across several image classification datasets.

## Key Objectives

- **Evaluate FFT Inputs:** Does the FFT-transformed input enhance classification performance in shallow CNNs compared to spatial inputs?
- **Hybrid Input Testing:** Does combining FFT and spatial inputs provide complementary benefits, especially under dropout regularization?
- **Comparison with Pre-trained Models:** How do shallow CNNs with FFT inputs compare to deeper, pre-trained models like ResNet?
- **Dataset Sensitivity:** Which datasets benefit most from FFT, spatial, or hybrid inputs, and how does performance vary across different image types?

## Datasets

The experiments use the following image classification datasets:

- **CIFAR-10:** 60,000 images (32x32x3), 10 classes.
- **CIFAR-100:** 100 classes for broader distribution insights.
- **EuroSAT:** 27,000 satellite images (64x64x3), 10 classes.
- **MNIST:** 70,000 handwritten digits (28x28x1), 10 classes.
- **SVHN:** 600,000 street view house numbers (32x32x3), 10 classes.
- **MS COCO (classification subset):** General object classification, 80 classes resized to 128x128x3.

## Methodology

### Data Preprocessing

- **FFT Transformation:** Apply FFT to each image's RGB channels separately and use the magnitude and phase components.
- **Hybrid Inputs:** Concatenate FFT and spatial inputs along the channel axis.
- **Normalization:** Normalize both FFT and spatial inputs to ensure stable training.

### Experimental Design

1. **Baseline Shallow CNN with Spatial Inputs:**
   - Architecture: 4 convolutional layers with max pooling and dropout.
   - Purpose: Establish baseline performance with spatial inputs.

2. **Shallow CNN with FFT Inputs:**
   - Architecture: Same as baseline, but with FFT inputs.
   - Purpose: Evaluate FFT input performance in shallow networks.

3. **Shallow CNN with Hybrid Inputs:**
   - Architecture: Dual input branches (FFT and spatial) merging into fully connected layers.
   - Purpose: Assess potential improvements from combined inputs.

4. **Fine-tuned Pre-trained ResNet:**
   - Fine-tuning only the final few layers.
   - Purpose: Compare FFT performance in shallow CNNs vs. deeper pre-trained models.

## Metrics of Evaluation

- **Accuracy:** Ratio of correct predictions.
- **Precision & Recall:** Especially important in multi-class datasets like MS COCO.
- **F1 Score:** Harmonic mean of precision and recall, useful in imbalanced datasets.
- **Training Time:** Total time for model training.
- **Convergence:** Number of epochs until validation accuracy stabilizes.
- **Failure Case Analysis:** Identifying scenarios where FFT inputs underperform, particularly on simpler datasets like MNIST.

## Timeline

- **Weeks 1-3:** Set up, load datasets, and run baseline experiments.
- **Weeks 4-6:** Evaluate shallow CNNs with FFT inputs.
- **Weeks 7-9:** Conduct hybrid input experiments.
- **Weeks 10-12:** Fine-tune the pre-trained ResNet.
- **Weeks 13-14:** Conduct robustness and generalization tests.
- **Weeks 15-16:** Finalize the report and documentation.

## Deliverables

1. **Performance Report:** A detailed analysis comparing FFT, spatial, and hybrid inputs.
2. **Failure Case Analysis:** Insights into failure scenarios for FFT inputs.
3. **Comparison of Shallow and Pre-trained Models:** Performance insights across model types.
4. **Robustness Report:** Performance under noise and data augmentations.
5. **Codebase:** Full Keras-based code for preprocessing, model architecture, and experiments.
6. **Final Report:** Complete with visualizations and recommendations for future work.

## License

This project is licensed under the MIT License.
