import numpy as np
import matplotlib.pyplot as plt

from models.simple_cnn import SimpleCNN
from layers.softmax_loss import softmax

# ---------- Test Function ----------
def test_sample_forward():
    model = SimpleCNN()

    # Generate random 28x28 input image
    X = np.random.rand(1, 1, 28, 28)

    # ---------- SHOW INPUT IMAGE ----------
    plt.imshow(X[0, 0], cmap='gray')
    plt.title("Input Image")
    plt.colorbar()
    plt.show()

    # ---------- FORWARD PASS ----------
    logits = model.forward(X)
    probs = softmax(logits)

    print("Logits:", logits)
    print("Probabilities:", probs)
    print("Predicted class:", np.argmax(probs))

# ---------- Run Test ----------
if __name__ == "__main__":
    test_sample_forward()
