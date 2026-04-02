# Deep Learning — ANN, CNN & Gradient Descent

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

A comprehensive collection of **deep learning implementations** spanning Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN), with real-world applications in customer churn prediction, heart failure prediction, and image classification.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Notebooks Overview](#notebooks-overview)
3. [Datasets](#datasets)
4. [Architecture 1: Artificial Neural Network (ANN)](#architecture-1-artificial-neural-network-ann)
   - [Theory](#ann-theory)
   - [Mathematical Formulation](#ann-math)
   - [Activation Functions](#activation-functions)
   - [ANN: Bank Churn Prediction](#ann-bank-churn-prediction)
   - [ANN: Heart Failure Prediction](#ann-heart-failure-prediction)
5. [Architecture 2: Gradient Descent — Simple NN](#architecture-2-gradient-descent--simple-nn)
6. [Architecture 3: Convolutional Neural Network (CNN)](#architecture-3-convolutional-neural-network-cnn)
   - [Theory](#cnn-theory)
   - [CNN: Cat vs Dog Classification](#cnn-cat-vs-dog-classification)
   - [CNN: Rock Paper Scissors Detection](#cnn-rock-paper-scissors-detection)
7. [Model Comparison](#model-comparison)
8. [Tech Stack](#tech-stack)
9. [Getting Started](#getting-started)
10. [Key Concepts Glossary](#key-concepts-glossary)
11. [References](#references)

---

## Repository Structure

```
Deep-Learning/
├── README.md
├── Artificial Neural Networks.ipynb            ← ANN: Bank churn prediction
├── Heart Failure Prediction ANN.ipynb          ← ANN: Medical classification
├── Gradient Descent - Simple NN.ipynb          ← Gradient descent fundamentals
├── Convolutional NN-Cat_Dog_Classification.ipynb  ← CNN: Binary image classification
├── CNN_Rock_Paper_Scissor_Detection.ipynb      ← CNN: Multi-class gesture detection
├── Churn_Modelling.csv                         ← Bank customer churn dataset
└── heart_failure_clinical_records_dataset.csv  ← Heart failure clinical data
```

| Notebook | Architecture | Task | Dataset |
|----------|-------------|------|---------|
| `Artificial Neural Networks.ipynb` | ANN (Dense) | Binary classification | `Churn_Modelling.csv` |
| `Heart Failure Prediction ANN.ipynb` | ANN (Dense) | Binary classification | `heart_failure_clinical_records_dataset.csv` |
| `Gradient Descent - Simple NN.ipynb` | Simple NN | Regression / demo | Synthetic |
| `Convolutional NN-Cat_Dog_Classification.ipynb` | CNN | Binary image classification | Cat/Dog images |
| `CNN_Rock_Paper_Scissor_Detection.ipynb` | CNN | Multi-class classification | RPS hand gestures |

---

## Datasets

### Churn_Modelling.csv

| Property | Value |
|----------|-------|
| Rows | 10,000 bank customers |
| Features | 14 (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary) |
| Target | `Exited` (1 = churned, 0 = stayed) |
| Churn rate | ~20% |
| Source | Kaggle — Bank Customer Churn Prediction |

### heart_failure_clinical_records_dataset.csv

| Property | Value |
|----------|-------|
| Rows | 299 patients |
| Features | 12 clinical features (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time) |
| Target | `DEATH_EVENT` (1 = died, 0 = survived) |
| Death rate | ~32% |
| Source | Kaggle / UCI — Heart Failure Clinical Records |

---

## Architecture 1: Artificial Neural Network (ANN)

### ANN Theory

An ANN is a computational model inspired by the biological brain, composed of layers of interconnected nodes (neurons). Each neuron computes a weighted sum of its inputs, applies a non-linear activation function, and passes the result to the next layer.

**Architecture for classification:**

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
(n features)    (units, ReLU)     (units, ReLU)     (1 unit, Sigmoid)
```

### ANN Math

**Forward pass through a single neuron:**

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

$$a^{(l)} = g^{(l)}\!\left(z^{(l)}\right)$$

where $W^{(l)}$ are weights, $b^{(l)}$ is the bias vector, and $g^{(l)}$ is the activation function at layer $l$.

**Loss function (Binary Cross-Entropy):**

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\hat{y}^{(i)} + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right]$$

**Backpropagation (gradient of loss w.r.t. weights):**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{m} \delta^{(l)} (a^{(l-1)})^T$$

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot g'^{(l)}(z^{(l)})$$

**Gradient descent weight update:**

$$W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial \mathcal{L}}{\partial W^{(l)}}$$

where $\alpha$ is the learning rate.

### Activation Functions

| Function | Formula | Range | Used In |
|----------|---------|-------|---------|
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | Hidden layers |
| **Sigmoid** | $\sigma(z) = 1/(1+e^{-z})$ | $(0, 1)$ | Binary output layer |
| **Softmax** | $e^{z_i}/\sum_j e^{z_j}$ | $(0, 1)$, sums to 1 | Multi-class output layer |
| **Tanh** | $(e^z-e^{-z})/(e^z+e^{-z})$ | $(-1, 1)$ | Hidden layers (older) |

---

### ANN: Bank Churn Prediction

**Goal:** Predict which customers will leave the bank next month.

**Preprocessing:**
```python
# Encode Geography (France/Spain/Germany) and Gender (Male/Female)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Label encode Gender
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# OneHot encode Geography
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
```

**Model Architecture:**

| Layer | Type | Units | Activation |
|-------|------|-------|------------|
| Input | Dense | 11 features | — |
| Hidden 1 | Dense | 6 | ReLU |
| Hidden 2 | Dense | 6 | ReLU |
| Output | Dense | 1 | Sigmoid |

```python
import tensorflow as tf

# Build ANN
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile
ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Evaluate
y_pred = (ann.predict(X_test) > 0.5).astype(int)
```

**Model Parameters:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Batch size | 32 |
| Epochs | 100 |
| Threshold | 0.5 |

---

### ANN: Heart Failure Prediction

**Goal:** Predict patient mortality from heart failure using 12 clinical features.

**Architecture:** Similar sequential ANN, adapted for the medical dataset with 12 input features → 2 hidden Dense layers with ReLU → 1 sigmoid output → Binary cross-entropy loss.

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = dataset.drop('DEATH_EVENT', axis=1).values
y = dataset['DEATH_EVENT'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1)
```

---

## Architecture 2: Gradient Descent — Simple NN

**Goal:** Visualize how gradient descent optimizes a simple neural network, building intuition for backpropagation.

**Key concepts demonstrated:**
- Forward propagation (compute predictions)
- Loss calculation (MSE)
- Backward propagation (compute gradients)
- Weight update via gradient descent

$$\mathcal{L}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

$$\theta := \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$

The notebook visually traces how weights evolve epoch-by-epoch as the loss decreases.

---

## Architecture 3: Convolutional Neural Network (CNN)

### CNN Theory

CNNs are specialized for processing grid-structured data (images). They apply learned **convolutional filters** that detect spatial features (edges, textures, shapes) regardless of position.

**Key layers:**

| Layer | Operation | Purpose |
|-------|-----------|---------|
| **Conv2D** | Convolution with learnable filters | Extract spatial features (edges, patterns) |
| **MaxPooling2D** | Take max value in each pool window | Downsample, reduce spatial size |
| **Flatten** | Reshape 3D → 1D | Connect to Dense layers |
| **Dense** | Fully connected | Classification decision |
| **Dropout** | Randomly zero activations | Regularization against overfitting |

**Convolution operation:**

$$(f * g)[n, m] = \sum_{i} \sum_{j} f[i, j] \cdot g[n-i, m-j]$$

Feature map size after convolution:
$$\text{out\_size} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

where $n$ = input size, $p$ = padding, $k$ = kernel size, $s$ = stride.

---

### CNN: Cat vs Dog Classification

**Goal:** Binary image classification — cat (0) or dog (1).

```
Input (64×64×3 RGB)
  ↓
Conv2D(32, 3×3, ReLU)  →  MaxPooling2D(2×2)
  ↓
Conv2D(32, 3×3, ReLU)  →  MaxPooling2D(2×2)
  ↓
Flatten
  ↓
Dense(128, ReLU)
  ↓
Dense(1, Sigmoid)         ← Binary output
```

```python
import tensorflow as tf

# Data augmentation to prevent overfitting
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary'
)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary'
)

# Build CNN
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
```

| Hyperparameter | Value |
|----------------|-------|
| Input size | 64 × 64 × 3 |
| Filters | 32 per Conv2D layer |
| Kernel size | 3 × 3 |
| Pool size | 2 × 2 |
| Dense units | 128 |
| Epochs | 25 |
| Loss | Binary cross-entropy |

---

### CNN: Rock Paper Scissors Detection

**Goal:** Multi-class classification of hand gestures — Rock (0), Paper (1), or Scissors (2).

Same CNN architecture as Cat/Dog, but with `class_mode='categorical'` and `softmax` output:

```python
# Output layer for multi-class
tf.keras.layers.Dense(units=3, activation='softmax')  # 3 classes

# Loss function for multi-class
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## Model Comparison

| Model | Task | Architecture | Key Layers | Optimizer | Loss | Accuracy |
|-------|------|-------------|-----------|-----------|------|----------|
| ANN (Churn) | Binary classification | 3-layer Dense | Dense(6), Dense(6), Dense(1, sigmoid) | Adam | Binary CE | ~86% |
| ANN (Heart Failure) | Binary classification | 4-layer Dense + Dropout | Dense(64), Dropout(0.3), Dense(32), Dense(1) | Adam | Binary CE | ~80% |
| CNN (Cat/Dog) | Binary image classification | 2×Conv+Pool + Dense | Conv2D(32), MaxPool2D, Dense(128) | Adam | Binary CE | ~80% |
| CNN (RPS) | Multi-class image | 2×Conv+Pool + Dense | Conv2D(32), MaxPool2D, Dense(3, softmax) | Adam | Categorical CE | ~95% |
| Simple NN | Regression/demo | Manual | Single hidden layer | Gradient descent | MSE | — |

---

## Tech Stack

| Library | Version | Usage |
|---------|---------|-------|
| Python | 3.x | Core language |
| TensorFlow | 2.x | Deep learning framework |
| Keras | 2.x (via TF) | High-level ANN/CNN building API |
| NumPy | 1.x | Array operations |
| Pandas | 1.x | Tabular data loading |
| scikit-learn | 1.x | Preprocessing, metrics, train-test split |
| Matplotlib | 3.x | Training curves, confusion matrix |
| Jupyter | Latest | Interactive notebook environment |

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/nithinrajkore/Deep-Learning.git
cd Deep-Learning

# 2. Install dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib jupyter

# 3. Run ANN — Bank Churn
jupyter notebook "Artificial Neural Networks.ipynb"

# 4. Run ANN — Heart Failure
jupyter notebook "Heart Failure Prediction ANN.ipynb"

# 5. Run CNN — Cat vs Dog
jupyter notebook "Convolutional NN-Cat_Dog_Classification.ipynb"

# 6. Run CNN — Rock Paper Scissors
jupyter notebook "CNN_Rock_Paper_Scissor_Detection.ipynb"
```

---

## Key Concepts Glossary

| Term | Definition |
|------|------------|
| **Neuron** | Computational unit: weighted sum of inputs → activation function → output |
| **Layer** | Collection of neurons processing input simultaneously |
| **Activation Function** | Non-linear function applied to neurons output (ReLU, Sigmoid, Softmax) |
| **Forward Propagation** | Computing predictions by passing inputs through the network |
| **Backpropagation** | Computing gradients of loss w.r.t. weights via the chain rule |
| **Gradient Descent** | Optimization algorithm that updates weights in the direction of steepest loss descent |
| **Adam** | Adaptive Moment Estimation optimizer — combines momentum and RMSprop |
| **Batch Size** | Number of samples processed per gradient update |
| **Epoch** | One full pass through the entire training dataset |
| **Dropout** | Regularization: randomly deactivates neurons during training to prevent overfitting |
| **Convolution** | Sliding a learnable filter over an image to detect local spatial features |
| **Pooling** | Downsampling operation that reduces spatial dimensions |
| **Feature Map** | Output of a convolutional layer — spatial activation pattern for a filter |
| **Binary Cross-Entropy** | Loss for binary classification: $-y\log\hat{y}-(1-y)\log(1-\hat{y})$ |
| **Categorical Cross-Entropy** | Loss for multi-class: $-\sum_k y_k \log \hat{y}_k$ |
| **Data Augmentation** | Artificially expand training data via rotations, flips, zooms |

---

## References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep Learning*. Nature, 521, 436–444.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning.
4. TensorFlow/Keras: [https://www.tensorflow.org/api_docs/python/tf/keras](https://www.tensorflow.org/api_docs/python/tf/keras)
5. Dataset (Churn): [Kaggle — Churn Modelling](https://www.kaggle.com/)
6. Dataset (Heart Failure): [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
