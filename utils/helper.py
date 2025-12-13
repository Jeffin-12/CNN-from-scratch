import numpy as np

def accuracy(pred, y):
    return np.mean(np.argmax(pred, axis=1) == y)
 
