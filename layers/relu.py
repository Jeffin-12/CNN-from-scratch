import numpy as np

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self.x > 0)
 
