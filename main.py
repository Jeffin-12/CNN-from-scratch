import numpy as np
from models.simple_cnn import SimpleCNN
from layers.softmax_loss import softmax, cross_entropy, softmax_backward
from data.mnist_loader import load_mnist
from config import LEARNING_RATE, EPOCHS, BATCH_SIZE

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Create model (DO NOT load here)
model = SimpleCNN()

num_samples = X_train.shape[0]
num_batches = num_samples // BATCH_SIZE

print("Training started...\n")

for epoch in range(EPOCHS):
    indices = np.random.permutation(num_samples)
    X_train = X_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0

    for i in range(num_batches):
        X_batch = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        logits = model.forward(X_batch)
        probs = softmax(logits)
        loss = cross_entropy(probs, y_batch)
        epoch_loss += loss

        grad = softmax_backward(probs, y_batch)
        model.backward(grad, LEARNING_RATE)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/num_batches:.4f}")

# ✅ SAVE AFTER TRAINING
model.save("checkpoints/cnn_weights.pkl")
print("✅ Model saved to checkpoints/cnn_weights.pkl")
