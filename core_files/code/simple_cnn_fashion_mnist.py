import subprocess
import sys

# Dictionary mapping module names to their pip install package names
required_packages = {
    'numpy': 'numpy',
    'tensorflow': 'tensorflow',
    'cv2': 'opencv-python',       # 'cv2' comes from opencv-python
    'pywt': 'PyWavelets',         # 'pywt' comes from PyWavelets
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',    # scikit-learn provides the 'sklearn' module
    'pandas': 'pandas'
}

def install_package(package):
    """Installs a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check for each package and install if missing
for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
        print(f"{module_name} is already installed.")
    except ImportError:
        print(f"{module_name} is not installed. Installing {package_name}...")
        install_package(package_name)

print("All required packages are installed.")

import numpy as np
import tensorflow as tf
import cv2
import pywt
import logging
import matplotlib.pyplot as plt
from scipy.fftpack import fft2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize grayscale images and reshape
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (num_samples, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=-1)
logging.info("Images preprocessed.")

# ------------- 1. Compute FFT Features (Real + Imaginary as 2 Channels) -------------
def compute_fft_image(images):
    fft_images = []
    for img in images:
        fft_img = fft2(img[..., 0])  # FFT on the single grayscale channel
        fft_real = abs(np.real(fft_img))  # Magnitude part
        fft_imag = np.angle(np.imag(fft_img))  # Phase part
        fft_image = np.stack([fft_real, fft_imag], axis=-1)  # Shape: (28, 28, 2)
        fft_images.append(fft_image)
    return np.array(fft_images)

# ------------- 2. Compute Wavelet Features (4 Channels) -------------
def compute_wavelet_image(images, wavelet='haar', level=1):
    wavelet_images = np.empty((images.shape[0], 14, 14, 4), dtype=np.float16)  # Pre-allocate
    for i, img in enumerate(images):
        coeffs2 = pywt.wavedec2(img[..., 0], wavelet, level=level)  # Wavelet on single channel
        cA, (cH, cV, cD) = coeffs2  # Extract coefficients
        cA, cH, cV, cD = [cv2.resize(c, (14, 14), interpolation=cv2.INTER_LINEAR) for c in [cA, cH, cV, cD]]
        wavelet_images[i] = np.stack([cA, cH, cV, cD], axis=-1)
    return wavelet_images

# ------------- 3. Define CNN Model -------------
def build_cnn(input_shape, num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('fashion_mnist_best_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# Model Evaluation Function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test).argmax(axis=1)
    accuracy = np.mean(y_pred == y_test)
    logging.info(f"{model_name} Test Accuracy: {accuracy:.6f}")
    print(f"=== Classification Report for {model_name} ===")
    print(classification_report(y_test, y_pred))
    print(f"=== Confusion Matrix for {model_name} ===")
    print(confusion_matrix(y_test, y_pred))

# Train CNN on Original MNIST
cnn_mnist = build_cnn(input_shape=(28, 28, 1))
cnn_mnist.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks, verbose=1)
cnn_mnist.save("cnn_fashion_mnist_model.keras")
evaluate_model(cnn_mnist, X_test, y_test, "Original MNIST")

# Train CNN on FFT Transformed Images
X_train_fft = compute_fft_image(X_train)
X_test_fft = compute_fft_image(X_test)
cnn_fft = build_cnn(input_shape=(28, 28, 2))
cnn_fft.fit(X_train_fft, y_train, epochs=100, validation_split=0.2, callbacks=callbacks, verbose=1)
cnn_fft.save("cnn_fft_fashion_mnist_model.keras")
evaluate_model(cnn_fft, X_test_fft, y_test, "FFT MNIST")

# Train CNN on Wavelet Transformed Images
X_train_wavelet = compute_wavelet_image(X_train)
X_test_wavelet = compute_wavelet_image(X_test)
cnn_wavelet = build_cnn(input_shape=(14, 14, 4))
cnn_wavelet.fit(X_train_wavelet, y_train, epochs=100, validation_split=0.2, callbacks=callbacks, verbose=1)
cnn_wavelet.save("cnn_wavelet_fashion_mnist_model.keras")
evaluate_model(cnn_wavelet, X_test_wavelet, y_test, "Wavelet MNIST")


import numpy as np
import tensorflow as tf
import cv2
import pywt
import logging
import matplotlib.pyplot as plt
from scipy.fftpack import fft2
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize grayscale images and reshape
X_test = X_test.astype("float32") / 255.0
X_test = np.expand_dims(X_test, axis=-1)  # Shape: (num_samples, 28, 28, 1)
logging.info("Images preprocessed.")

# Define percentages of missing data
mask_percentages = range(0,100,1)#[0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

# Function to apply random masking
def apply_random_masking(images, percentage):
    masked_images = images.copy()
    mask = np.random.rand(*images.shape[:3]) < (percentage / 100.0)
    masked_images[mask, :] = 0
    return masked_images

# Function to compute FFT representation
def compute_fft_image(images):
    fft_images = []
    for img in images:
        fft_img = fft2(img[..., 0])  # FFT on the single grayscale channel
        fft_real = abs(np.real(fft_img))  # Magnitude part
        fft_imag = np.angle(np.imag(fft_img))  # Phase part
        fft_image = np.stack([fft_real, fft_imag], axis=-1)  # Shape: (28, 28, 2)
        fft_images.append(fft_image)
    return np.array(fft_images)

# Function to compute wavelet representation
def compute_wavelet_image(images, wavelet='haar', level=1):
    wavelet_images = np.empty((images.shape[0], 14, 14, 4), dtype=np.float16)  # Pre-allocate
    for i, img in enumerate(images):
        coeffs2 = pywt.wavedec2(img[..., 0], wavelet, level=level)  # Wavelet on single channel
        cA, (cH, cV, cD) = coeffs2  # Extract coefficients
        cA, cH, cV, cD = [cv2.resize(c, (14, 14), interpolation=cv2.INTER_LINEAR) for c in [cA, cH, cV, cD]]
        wavelet_images[i] = np.stack([cA, cH, cV, cD], axis=-1)
    return wavelet_images

# Load pretrained models
#cnn_mnist = load_model("cnn_mnist_model.keras")
#cnn_fft = load_model("cnn_fft_mnist_model.keras")
#cnn_wavelet = load_model("cnn_wavelet_mnist_model.keras")

# Dictionary to store accuracy results
accuracy_results = {
    "Spatial": [],
    "FFT": [],
    "Wavelet": []
}

# Output directories
os.makedirs("simple_cnn_fashion_mnist_confusion_matrices", exist_ok=True)
os.makedirs("simple_cnn_fashion_mnist_correlation_plots", exist_ok=True)
os.makedirs("simple_cnn_fashion_mnist_predition_output", exist_ok=True)

# Evaluate robustness for different masking percentages
for p in mask_percentages:
    logging.info(f"Evaluating for {p}% missing data")

    # Apply masking
    X_test_masked = apply_random_masking(X_test, p)
    X_test_fft_masked = compute_fft_image(X_test_masked)
    X_test_wavelet_masked = compute_wavelet_image(X_test_masked)

    # Function to evaluate model and save confusion matrix
    def evaluate_and_save(model, X_test_transformed, name, p):
        y_pred = model.predict(X_test_transformed).argmax(axis=1)
        accuracy = np.mean(y_pred == y_test)
        accuracy_results[name].append(accuracy)

        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
        df_cm.to_csv(f"simple_cnn_fashion_mnist_confusion_matrices/{name}_confusion_{p}.csv")
        df_prediction = pd.DataFrame()
        df_prediction['True Label'] = y_test
        df_prediction['Predicted Label'] = y_pred
        df_prediction.to_csv(f"simple_cnn_fashion_mnist_predition_output/{name}_predictions_{p}.csv")

        return accuracy, df_cm

    # Evaluate all models
    acc_spatial, cm_spatial = evaluate_and_save(cnn_mnist, X_test_masked, "Spatial", p)
    acc_fft, cm_fft = evaluate_and_save(cnn_fft, X_test_fft_masked, "FFT", p)
    acc_wavelet, cm_wavelet = evaluate_and_save(cnn_wavelet, X_test_wavelet_masked, "Wavelet", p)

    # Generate correlation plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm_spatial.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0])
    axes[0].set_title(f'Spatial - {p}% Data Loss')

    sns.heatmap(cm_fft.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1])
    axes[1].set_title(f'FFT - {p}% Data Loss')

    sns.heatmap(cm_wavelet.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2])
    axes[2].set_title(f'Wavelet - {p}% Data Loss')

    plt.savefig(f"simple_cnn_fashion_mnist_correlation_plots/correlation_plot_{p}.png")
    plt.close()

df_accuracy = pd.DataFrame(accuracy_results, index=range(0, 100, 1))
df_accuracy.to_csv("simple_cnn_fashion_mnist_predition_output/accuracy_results.csv", index_label="Data Loss %")
plt.figure(figsize=(10, 6))
for column in df_accuracy.columns:
    plt.plot(df_accuracy.index, df_accuracy[column], marker='o', label=column)

# Formatting the plot
plt.xlabel("Percentage Data Loss")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Percentage Data Loss")
plt.legend()
plt.grid(True)
plt.ylim((0,1.1))
# Save the plot
plt.savefig("simple_cnn_fashion_mnist_correlation_plots/Accuracy_vs_DataLoss.png")
plt.show()

# Compute ADR for each representation
adr_spatial = [(accuracy_results["Spatial"][0] - acc) / accuracy_results["Spatial"][0] * 100 for acc in accuracy_results["Spatial"]]
adr_fft = [(accuracy_results["FFT"][0] - acc) / accuracy_results["FFT"][0] * 100 for acc in accuracy_results["FFT"]]
adr_wavelet = [(accuracy_results["Wavelet"][0] - acc) / accuracy_results["Wavelet"][0] * 100 for acc in accuracy_results["Wavelet"]]

# Plot ADR vs Data Loss
plt.figure(figsize=(10, 6))
plt.plot(mask_percentages, adr_spatial, label="Spatial", marker='o')
plt.plot(mask_percentages, adr_fft, label="FFT", marker='s')
plt.plot(mask_percentages, adr_wavelet, label="Wavelet", marker='^')
plt.xlabel("Percentage Data Loss")
plt.ylabel("Accuracy Drop Rate (ADR) %")
plt.title("ADR vs. Percentage Data Loss")
plt.legend()
plt.grid(True)
plt.savefig("simple_cnn_fashion_mnist_ADR_vs_DataLoss.png")
plt.show()

logging.info("Analysis completed and all results saved.")

