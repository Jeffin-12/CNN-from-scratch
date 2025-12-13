import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.simple_cnn import SimpleCNN
from layers.softmax_loss import softmax


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))

    img_np = np.array(img, dtype=np.float32)
    img_np = 255.0 - img_np          # invert colors
    img_np /= 255.0                  # normalize

    return img_np.reshape(1, 1, 28, 28), img_np


def test_custom_image(image_path):
    model = SimpleCNN()
    model.load("checkpoints/cnn_weights.pkl")


    X, img_display = preprocess_image(image_path)

    plt.imshow(img_display, cmap="gray")
    plt.title("Custom Input Image")
    plt.axis("off")
    plt.show()

    logits = model.forward(X)
    probs = softmax(logits)

    print("Logits:", logits)
    print("Probabilities:", probs)
    print("Predicted class:", np.argmax(probs))


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "custom_digit_2.png")

    test_custom_image(image_path)
