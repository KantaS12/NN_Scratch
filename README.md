# Neural Network From Scratch

## Project Overview

This project implements a basic feedforward neural network from scratch using Python and the NumPy library. The network is trained and evaluated on the MNIST dataset, a widely used benchmark for handwritten digit recognition. The goal of this project is to provide a clear and understandable implementation of fundamental neural network concepts, including forward propagation, backpropagation, gradient descent, and optimization techniques. The project explores the impact of various hyperparameters, network architectures, and optimization algorithms on the model's performance.

## Features

* **Feedforward Neural Network:** Implements a multi-layer perceptron (MLP) architecture.
* **MNIST Dataset:** Utilizes the MNIST dataset for training and evaluation.
* **Variable Network Architecture:** Allows for customization of the number of hidden layers and the dimensionality of each layer.
* **Activation Functions:** Includes support for common activation functions (e.g., sigmoid, ReLU - though the provided conclusion doesn't explicitly mention which were used).
* **Loss Function:** Implements a suitable loss function for multi-class classification (likely cross-entropy loss).
* **Gradient Descent:** Implements standard gradient descent for model training.
* **Adam Optimizer:** Includes an implementation of the Adam optimization algorithm.
* **Learning Rate Exploration:** Demonstrates the impact of different learning rates on training.
* **Performance Evaluation:** Calculates and reports training and testing accuracy and loss.
* **Visualization:** (Potentially) Includes plotting of training loss and accuracy over iterations.

## Requirements

* Python 3.x
* NumPy (>= 1.20.0)
* Pandas (>= 1.3.0)
* Matplotlib (>= 3.4.0)

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib
```

## Installation

1. Clone Repository
```bash
git clone <https://github.com/KantaS12/NN_Scratch>
```

2. Download MNIST DataSet
I used kaggle note book but you can also use a kaggle dataset or download using libraries.

## Using Example

```python
def init_params():
    W1 = np.random.uniform(-0.3, 0.3, (32, 784))
    b1 = np.random.uniform(-0.3, 0.3, (32, 1))
    
    W2 = np.random.uniform(-0.3, 0.3, (32, 32))
    b2 = np.random.uniform(-0.3, 0.3, (32, 1))
                        
    W3 = np.random.uniform(-0.3, 0.3, (64, 32))
    b3 = np.random.uniform(-0.3, 0.3, (64, 1))

    W4 = np.random.uniform(-0.3, 0.3, (64, 64))
    b4 = np.random.uniform(-0.3, 0.3, (64, 1))

    W5 = np.random.uniform(-0.3, 0.3, (10, 64))
    b5 = np.random.uniform(-0.3, 0.3, (10, 1))

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

def ReLU(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1/ (1 + np.exp(-Z))

def SeLU(Z, alpha=1.67326324, scale=1.05070098):
    return np.where(Z > 0, scale * Z, scale * alpha * (np.exp(Z) - 1))

def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def ReLU_deriv(Z):
    return Z > 0

def sigmoid_deriv(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def SeLU_deriv(Z, alpha=1.67326324, scale=1.05070098):
    return np.where(Z > 0, scale, scale * alpha * np.exp(Z))
    
def tanh_deriv(Z):
    return 1 - (tanh(Z) ** 2)

def one_hot(Y): #Make everything 0's except the most "valued" value!
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    
    return one_hot_Y

def forward_prop(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5):
    Z1 = W1.dot(X) + b1 #(32, 784) * (784, m) + (32,1)
    A1 = ReLU(Z1) #(32, m) 
    
    Z2 = W2.dot(A1) + b2 #(32, 32) @ (32, m) + (32, 1)
    A2 = sigmoid(Z2) #(32, m)
    
    Z3 = W3.dot(A2) + b3 #(64, 32) @ (32, m) + (64, 1)
    A3 = SeLU(Z3) #(64, m)
    
    Z4 = W4.dot(A3) + b4 #(64, 64) @ (64, m) + (64, 1)
    A4 = tanh(Z4) #(64, m)
    
    Z5 = W5.dot(A4) + b5 #(10, 64) @ (64, m) + (10, 1)
    A5 = softmax(Z5) #(10, m)
    
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5

def back_prop(X, Y, W1, W2, W3, W4, W5, Z1, Z2, Z3, Z4, Z5, A1, A2, A3, A4, A5):
    one_hot_Y = one_hot(Y)
    
    dZ5 = A5 - one_hot_Y
    dW5 = 1 / m * dZ5.dot(A4.T)
    db5 = 1 / m * np.sum(dZ5, axis=1, keepdims=True)
    
    dZ4 = W5.T.dot(dZ5) * tanh_deriv(Z4)
    dW4 = 1 / m * dZ4.dot(A3.T)
    db4 = 1 / m * np.sum(dZ4, axis=1, keepdims=True)

    dZ3 = W4.T.dot(dZ4) * SeLU_deriv(Z3)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * sigmoid_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5

def update_params(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    
    W4 = W4 - alpha * dW4
    b4 = b4 - alpha * db4

    W5 = W5 - alpha * dW5
    b5 = b5 - alpha * db5

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = init_params()
    history = {'loss': [], 'accuracy': []}
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5 = forward_prop(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
        dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5 = back_prop(X, Y, W1, W2, W3, W4, W5, Z1, Z2, Z3, Z4, Z5, A1, A2, A3, A4, A5)
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = update_params(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, alpha)
        if i % 10 == 0:
            predictions = np.argmax(A5, 0)
            accuracy = get_accuracy(predictions, Y)
            one_hot_Y = one_hot(Y)
            loss = cross_entropy_loss(A5, one_hot_Y)
            print(f"Iteration: {i}, Accuracy: {accuracy:.5f}, Loss: {loss:.5f}")
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, iterations, 10), history['loss'])
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy Loss")
    plt.title("Training Loss")

    plt.subplot(1,2, 2)
    plt.plot(range(0, iterations, 10), history['accuracy'])
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.tight_layout()
    plt.show()
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5
```

## Implementation Details / Neural Network Methods

### Initialization: 
Initializing the weights and biases of the network layers (e.g., using random initialization).

### Forward Propagation
Implementing the forward pass through the network, calculating the output of each layer by applying matrix multiplications, adding biases, and applying activation functions.

### Backward Propagation
Implementing the backpropagation algorithm to calculate the gradients of the loss function with respect to the network's weights and biases. This involves applying the chain rule of calculus.

### Loss Function
Implementing a function to calculate the loss (e.g., cross-entropy loss for multi-class classification).

### Prediction
Making predictions based on the output of the forward pass (e.g., selecting the class with the highest probability).

### Training
Implementing the training loop, which iterates over the training data, performs forward and backward propagation, and updates the weights and biases using the chosen optimization algorithm.

### Evaluation
Calculating performance metrics such as accuracy and loss on the test set.

## Results and Findings

### Overfitting
The high training accuracy (98%) and low training loss (0.07) strongly suggest overfitting. This indicates that the model has learned the training data too well and may not generalize well to unseen data.

### Learning Rate Impact
Different learning rates (0.01, 0.1, and others explored) had varying effects on the training process. A learning rate of 0.001 was found to yield the most favorable training outcome, 
likely striking a better balance between convergence speed and stability.

### Network Architecture
The dimensionality of the input, hidden, and output layers significantly influenced the results. The specific architecture used (number of hidden layers and their sizes) plays a crucial role in the model's capacity and ability to learn complex patterns.

### Impact of Training Iterations
Increasing the number of training iterations contributed to the observed overfitting, as the model had more opportunities to memorize the training data.

### Adam Optimizer Effectiveness
The Adam optimizer achieved a significantly higher testing accuracy (98%) compared to standard gradient descent (89%), representing a notable 9% improvement. 
This highlights the effectiveness of adaptive learning rate methods like Adam in enhancing the optimization process, potentially by better navigating the loss landscape and converging faster to a better solution. 
The improvement was substantial even when considering the impact of overfitting and architectural choices.

### Testing Accuracy
The final testing accuracy achieved by the model using the Adam optimizer was 0.9538095238095238 (approximately 95.38%).

## Future Improvements

### Regularization Techniques
Incorporate regularization techniques such as L1 or L2 regularization, or dropout, to mitigate overfitting and improve generalization.

### Batch Normalization
Add batch normalization layers to improve training stability and potentially allow for the use of higher learning rates.

### More Sophisticated Architectures
Explore more complex network architectures, such as convolutional neural networks (CNNs), which are known to perform well on image recognition tasks.

### Hyperparameter Tuning 
Implement more systematic hyperparameter tuning techniques (e.g., grid search, random search) to find the optimal learning rate, network architecture, and other hyperparameters.

### Early Stopping
Implement early stopping to prevent overfitting by monitoring the performance on a validation set and stopping training when the performance starts to degrade.

### Data Augmentation
Apply data augmentation techniques to the MNIST dataset (e.g., rotations, small translations) to increase the size and diversity of the training data and improve generalization.

### Visualization Enhancements
Add more detailed visualizations of the training process, such as plots of weights and gradients.

## License

This work uses the MNIST database of handwritten digits, which is copyrighted by Yann LeCun (Courant Institute, NYU) and Corinna Cortes (Google Labs, New York) and made available under the CC BY-SA 3.0 license.

## Acknowledgments

The MNIST dataset is a widely used resource for the machine learning community.

My friend Bryan.

## Author 

Kanta Saito 
