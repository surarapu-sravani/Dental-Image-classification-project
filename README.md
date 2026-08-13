# Dental-Image-classification-project
final year project using Resnet50 with SVM
# Improved Integrated Deep Learning Framework with Support Vector Machine for Dental Health Analysis

## 📌 Project Overview

Dental health problems such as cavities, fractured teeth, and impacted teeth are common and require early detection. Traditionally, dental X-ray images are analyzed manually by dentists, which can be time-consuming and may be affected by human error.

This project presents an **improved deep learning-based framework for automated dental health analysis using dental X-ray images**.

The system classifies individual tooth images into two categories:

* **Healthy**
* **Non-Healthy** – includes cavities, fractured teeth, and impacted teeth

Different deep learning architectures were evaluated, including **CNN, VGG16, ResNet50, ResNet101, and DenseNet121**. A hybrid **ResNet50 + SVM** approach was then developed, where ResNet50 extracts high-level image features and an SVM classifier performs the final classification.

---

## 🎯 Objectives

* Automate the classification of dental X-ray images.
* Identify whether a tooth is Healthy or Non-Healthy.
* Compare the performance of different deep learning architectures.
* Extract meaningful dental image features using a pre-trained ResNet50 model.
* Improve classification using a hybrid **ResNet50 + SVM** approach.
* Evaluate the models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.

---

## 🗂️ Dataset

The project uses the **Dental Object Detection X-Ray Dataset** obtained from Kaggle.

### Dataset Details

* Original labeled panoramic dental X-ray images: **1,650**
* Tooth-level samples obtained after cropping: **6,937**
* Training samples: **5,474**
* Testing samples: **1,463**

The dataset was converted into a binary classification problem:

| Class       | Description                                         |
| ----------- | --------------------------------------------------- |
| Healthy     | Tooth without the considered dental conditions      |
| Non-Healthy | Tooth affected by cavities, fractures, or impaction |

The original X-ray images contain annotations for different dental conditions. Individual tooth regions were cropped from the labeled images to create tooth-level samples for classification.

> **Note:** The dataset itself is not included in this repository.

---

## 🔬 Methodology

The overall workflow consists of the following stages:

```text
Dental X-ray Dataset
        ↓
Data Loading & Preprocessing
        ↓
Tooth-level Image Cropping
        ↓
Train / Test Split
        ↓
Deep Learning Model Comparison
        ↓
ResNet50 Feature Extraction
        ↓
2048-Dimensional Feature Vector
        ↓
StandardScaler Normalization
        ↓
SVM Classifier (RBF Kernel)
        ↓
Healthy / Non-Healthy Prediction
        ↓
Performance Evaluation
```

---

## ⚙️ Preprocessing

The dental X-ray images were preprocessed before model training.

* Images resized to **224 × 224**
* Tooth-level regions extracted from annotated X-ray images
* Training and testing datasets prepared
* Deep features extracted using ResNet50
* Extracted features normalized using `StandardScaler`

---

## 🧠 Models Compared

The following models were evaluated:

### 1. CNN

A conventional Convolutional Neural Network was used as a baseline model. CNNs can automatically learn image features but may suffer from overfitting and limitations in capturing complex patterns.

### 2. VGG16

VGG16 was evaluated as a transfer-learning architecture. It provides a simple and effective architecture but has a large number of parameters and high memory requirements.

### 3. ResNet50

ResNet50 uses residual connections to enable deeper feature extraction while reducing the vanishing-gradient problem.

### 4. ResNet101

ResNet101 provides a deeper architecture capable of learning more complex image features, but it requires greater computational resources and training time.

### 5. DenseNet121

DenseNet121 uses dense connections to efficiently reuse features between layers, although its architecture can be computationally complex.

### 6. Proposed ResNet50 + SVM

The proposed hybrid model combines:

**ResNet50 → Feature Extraction → StandardScaler → SVM**

ResNet50 is used to extract high-level visual features, while SVM replaces the conventional Softmax classification layer and provides the final Healthy / Non-Healthy prediction.

---

## 🏗️ Proposed Architecture

```text
Input Dental X-ray
       ↓
Image Preprocessing
       ↓
ResNet50
       ↓
Feature Extraction
       ↓
Global Average Pooling
       ↓
2048-Dimensional Features
       ↓
StandardScaler
       ↓
SVM
       ↓
 ┌───────────────┐
 │               │
Healthy      Non-Healthy
```

The implementation process described in the project uses ResNet50 as the feature extractor, Global Average Pooling to obtain **2048-dimensional features**, StandardScaler for normalization, and an **RBF-kernel SVM** for final classification.

---

## 📊 Model Performance

The following results were obtained during the comparative evaluation:

| Model              |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------ | ---------: | ---------: | ---------: | ---------: |
| CNN                |     79.63% |     77.43% |     85.07% |     81.07% |
| VGG16              |     79.29% |     77.90% |     83.22% |     80.46% |
| ResNet50           |     83.05% |     82.77% |     84.53% |     83.64% |
| ResNet101          |     82.16% |     81.79% |     83.87% |     82.83% |
| DenseNet121        |     77.31% |     80.65% |     73.33% |     76.82% |
| **ResNet50 + SVM** | **85.85%** | **85.87%** | **85.81%** | **85.83%** |

The proposed **ResNet50 + SVM** approach achieved the best overall performance among the models presented in the project.
## 📈 Evaluation

The proposed model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve

### Confusion Matrix

The confusion matrix for the ResNet50 + SVM model contains:

* **601** correctly classified Healthy samples
* **655** correctly classified Non-Healthy samples
* **112** Healthy samples classified as Non-Healthy
* **95** Non-Healthy samples classified as Healthy

The ROC curve was also generated to evaluate the classification performance across different decision thresholds.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Machine Learning

* Scikit-learn
* Support Vector Machine (SVM)

### Data Processing

* NumPy
* Pandas

### Image Processing / Visualization

* OpenCV
* Matplotlib
* Seaborn

### Development Platform

* Google Colab

### Hardware

* GPU acceleration using NVIDIA T4 in Google Colab

### 1. Install the required libraries
### 2. Open the notebook

The main implementation is provided as a Jupyter/Google Colab notebook.

```text
notebooks/dental_health_analysis.ipynb
```
The notebook can be opened directly in **Google Colab** for GPU-based execution.

### 3. Add the dataset
Download the required dental X-ray dataset separately and configure the dataset path in the notebook.

### 4. Run the notebook
Run the preprocessing, feature extraction, model training, and evaluation cells sequentially.
## 📋 Requirements

The project requires Python and the following major libraries:

```text
tensorflow
keras
scikit-learn
numpy
pandas
matplotlib
seaborn
opencv-python
```
## 🔍 Key Findings

The comparative experiments show that different deep learning architectures provide different levels of performance for dental X-ray classification.

The proposed hybrid approach combines the strong feature extraction capability of **ResNet50** with the classification capability of **SVM**.

Replacing the conventional Softmax-based classification approach with SVM provides a stronger decision boundary for the binary Healthy vs Non-Healthy classification task.

The ResNet50 + SVM model achieved:

* **85.85% Accuracy**
* **85.87% Precision**
* **85.81% Recall**
* **85.83% F1-Score**
## 🔮 Future Scope

The project can be further improved by:

* Increasing the dataset size to improve generalization.
* Extending the system to multi-class dental disease classification.
* Detecting individual conditions such as caries, fractures, and impacted teeth separately.
* Deploying the model as a web or mobile application.
* Implementing real-time dental X-ray analysis.
* Exploring ensemble learning techniques.
* Extending the system to additional oral diseases such as periodontitis and gum infections.


## ⭐ Project Highlights

* 🦷 Dental X-ray image classification
* 🤖 Deep learning-based analysis
* 🧠 ResNet50 feature extraction
* 📊 SVM-based classification
* 🔬 Comparison of multiple deep learning architectures
* 📈 Accuracy, Precision, Recall and F1-score evaluation
* 📉 Confusion Matrix and ROC Curve analysis
* ☁️ Developed using Google Colab
