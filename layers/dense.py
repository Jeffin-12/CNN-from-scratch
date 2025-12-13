import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        limit = np.sqrt(1.0 / input_size)
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros((1, output_size))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d_out, lr):
        dW = self.x.T @ d_out
        db = np.sum(d_out, axis=0, keepdims=True)
        d_x = d_out @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db

        return d_x
 
