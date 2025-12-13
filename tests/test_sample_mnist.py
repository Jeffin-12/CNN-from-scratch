import numpy as np
import matplotlib.pyplot as plt

from data.mnist_loader import load_mnist
from models.simple_cnn import SimpleCNN
from layers.softmax_loss import softmax


def test_mnist_image():
    # Load MNIST dataset
    X_train, y_train, _, _ = load_mnist()

    # Choose an index to display (change this to show different digits)
    idx = 0

    image = X_train[idx]  # shape: (1, 28, 28)
    label = y_train[idx]

    # ------- DISPLAY IMAGE -------
    plt.imshow(image[0], cmap="gray")
    plt.title(f"MNIST Image — Label: {label}")
    plt.colorbar()
    plt.show()

    # ------- RUN THROUGH MODEL -------
    model = SimpleCNN()

    image_batch = image.reshape(1, 1, 28, 28)
    logits = model.forward(image_batch)
    probs = softmax(logits)

    print("Label:", label)
    print("Logits:", logits)
    print("Probabilities:", probs)
    print("Predicted class:", np.argmax(probs))


if __name__ == "__main__":
    test_mnist_image()
