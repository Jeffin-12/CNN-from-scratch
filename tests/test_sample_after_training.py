import numpy as np
import matplotlib.pyplot as plt

from data.mnist_loader import load_mnist
from models.simple_cnn import SimpleCNN
from layers.softmax_loss import softmax, cross_entropy, softmax_backward
from utils.helper import accuracy
from config import LEARNING_RATE, EPOCHS, BATCH_SIZE


def test_model_after_training():
    # ---------------------------
    # Load MNIST dataset
    # ---------------------------
    X_train, y_train, X_test, y_test = load_mnist()

    model = SimpleCNN()

    num_samples = X_train.shape[0]
    num_batches = num_samples // BATCH_SIZE

    print("Training CNN...\n")

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(EPOCHS):
        # Shuffle dataset
        indices = np.random.permutation(num_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0

        for i in range(num_batches):
            X_batch = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            y_batch = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            logits = model.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy(probs, y_batch)
            epoch_loss += loss

            grad = softmax_backward(probs, y_batch)
            model.backward(grad, LEARNING_RATE)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / num_batches:.4f}")

    # ---------------------------
    # Testing the Model
    # ---------------------------
    print("\nEvaluating model on MNIST test set...")

    logits = model.forward(X_test)
    probs = softmax(logits)

    test_acc = accuracy(probs, y_test)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%\n")

    # ---------------------------
    # Show Sample Predictions
    # ---------------------------
    print("Showing 5 sample test images with predictions...")

    for i in range(5):
        img = X_test[i][0]

        plt.imshow(img, cmap="gray")
        plt.title(f"True: {y_test[i]}, Predicted: {np.argmax(probs[i])}")
        plt.show()


if __name__ == "__main__":
    test_model_after_training()
