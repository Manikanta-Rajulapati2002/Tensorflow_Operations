# README

## Overview
This repository contains multiple Jupyter Notebooks related to TensorFlow operations, model training, and optimization techniques. Below is a summary of each notebook:

### 1. Task_1.ipynb - Tensor Manipulations & Reshaping
- Covers various tensor operations, reshaping, and broadcasting in TensorFlow.
- Example tensor manipulation using `tf.random.uniform`.
- Demonstrates how to find rank and shape of tensors.

**Example Code Snippet:**
```python
import tensorflow as tf

tensor = tf.random.uniform(shape=(4,6), minval=1, maxval=20, dtype=tf.int32)
print("The random tensor is :", '\n', tensor.numpy())

# Reshaping the tensor into (2,3,4)
reshape_tensor = tf.reshape(tensor, (2, 3, 4))
rank = tf.rank(reshape_tensor)
shape = tf.shape(reshape_tensor)
```

---

### 2. Task_2.ipynb - Loss Functions & Hyperparameter Tuning
- Explores different types of loss functions used in neural networks.
- Demonstrates hyperparameter tuning techniques to optimize model performance.
- Uses TensorFlow and Keras for implementing loss functions and tuning.

**Example Code Snippet:**
```python
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy

# Define example tensors
y_true = tf.constant([1, 0, 0])
y_pred = tf.constant([0.8, 0.1, 0.1])

# Compute loss values
mse = MeanSquaredError()
ce = CategoricalCrossentropy()

print("Mean Squared Error:", mse(y_true, y_pred).numpy())
print("Categorical Crossentropy:", ce(y_true, y_pred).numpy())
```

---

### 3. Task_3.ipynb - Train a Model with Different Optimizers
- Implements training of an MNIST classifier using Adam and SGD optimizers.
- Uses TensorFlow's Keras API to define a neural network.
- Compares model performance with different optimizers.

**Example Code Snippet:**
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(optimizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

---

### 4. Task_4.ipynb - Train a Neural Network and Log to TensorBoard
- Trains an MNIST model while logging to TensorBoard.
- Includes analysis on training and validation accuracy.
- Demonstrates setting up logging with TensorBoard.

**Example Code Snippet:**
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
import datetime

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## Requirements
To run these notebooks, install the required dependencies using:
```bash
pip install tensorflow matplotlib
```

## Usage
Run each notebook in Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook
```

