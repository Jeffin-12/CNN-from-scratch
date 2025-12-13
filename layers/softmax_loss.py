import numpy as np

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(pred, y):
    batch = y.shape[0]
    return -np.sum(np.log(pred[range(batch), y])) / batch

def softmax_backward(pred, y):
    batch = y.shape[0]
    grad = pred.copy()
    grad[range(batch), y] -= 1
    grad /= batch
    return grad
 
