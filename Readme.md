# Neural Network

Simple implementation of a Nerual Network trained using Stochastic Gradient Descent with momentum.

## Supported python versions:
* Python 2.7
* Python 3.4

## Python package dependencies
* Numpy        (http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)

# Documentation

Setup model.

Following is an example of a classification problem using a network with 2 hidden layers of size 8 and 4 (following model parameters are default).

```python

from NeuralNetwork import NeuralNetwork
d = 5 # Dimension of input
K = 3 # Number of classes
layers = [d, 8, 4, K]
model = NeuralNetwork(layers=layers, num_epochs=1000, learning_rate=0.10,
                      alpha=0.9, activation_func='sigmoid', epsilon=0.001)
```


Train model

```python
model.fit(X, Y)
```

Predict new observations

```python
Y_hat = model.predict(X_test)
```
