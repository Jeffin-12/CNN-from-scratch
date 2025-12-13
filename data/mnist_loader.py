import numpy as np
from urllib.request import urlretrieve
import gzip

def load_mnist():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = "mnist.npz"
    urlretrieve(url, path)

    data = np.load(path)
    X_train = data['x_train'] / 255.0
    y_train = data['y_train']
    X_test = data['x_test'] / 255.0
    y_test = data['y_test']

    return X_train.reshape(-1, 1, 28, 28), y_train, X_test.reshape(-1, 1, 28, 28), y_test
 
