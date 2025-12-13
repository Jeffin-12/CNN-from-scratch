import pickle
import os
from layers.conv2d import Conv2D
from layers.relu import ReLU
from layers.maxpool import MaxPool2D
from layers.dense import Dense

class SimpleCNN:
    def __init__(self):
        self.conv = Conv2D(8, 3, 1)
        self.relu = ReLU()
        self.pool = MaxPool2D(2)
        self.fc = Dense(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.relu.forward(x)
        x = self.pool.forward(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc.forward(x)

    def backward(self, grad, lr):
        grad = self.fc.backward(grad, lr)
        grad = grad.reshape(-1, 8, 13, 13)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        self.conv.backward(grad, lr)

    # ---------- SAVE ----------
    def save(self, path="checkpoints/cnn_weights.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "conv_W": self.conv.W,
                "conv_b": self.conv.b,
                "fc_W": self.fc.W,
                "fc_b": self.fc.b
            }, f)

    # ---------- LOAD ----------
    def load(self, path="checkpoints/cnn_weights.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.conv.W = data["conv_W"]
            self.conv.b = data["conv_b"]
            self.fc.W = data["fc_W"]
            self.fc.b = data["fc_b"]
