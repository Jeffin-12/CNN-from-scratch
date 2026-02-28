
# 📘 Convolutional Neural Network From Scratch (NumPy Only)

*A minimal deep-learning framework implementing a full CNN without using PyTorch, TensorFlow, or Keras.*

---

## 🚀 Overview

This project implements a **Convolutional Neural Network (CNN) completely from scratch**, using only **NumPy**.  
No deep-learning libraries are used.

The goal is to understand the **internal mechanics** of deep learning, including:

- Convolution operations  
- Backpropagation through convolution  
- MaxPooling  
- Dense layers  
- Softmax + cross-entropy  
- Modular deep-learning architecture design  

This serves as a learning-focused implementation, ideal for students, engineers, and anyone studying ML at a deep level.

---

## 📂 Project Structure

```
cnn_from_scratch/
│
├── main.py                      # Training script / entry point
├── config.py                    # Hyperparameters
│
├── tests/
│   └── test_sample.py           # Test CNN with sample input
│
├── data/
│   └── mnist_loader.py          # MNIST dataset loader (NumPy)
│
├── layers/
│   ├── conv2d.py                # Convolution layer + backward
│   ├── relu.py                  # ReLU activation
│   ├── maxpool.py               # Max pooling + backward
│   ├── dense.py                 # Fully connected layer
│   └── softmax_loss.py          # Softmax + cross-entropy
│
├── models/
│   └── simple_cnn.py            # Full CNN architecture
│
└── utils/
    ├── helper.py                # Accuracy, helpers
    └── initializers.py          # Xavier weight initializer
```

---

## 🧠 Model Architecture

The default architecture is:

```
Input (1 × 28 × 28)
  ↓
Conv2D (8 filters, 3×3)
  ↓
ReLU
  ↓
MaxPool (2×2)
  ↓
Flatten (8 × 13 × 13)
  ↓
Dense Layer (1352 → 10)
  ↓
Softmax
```

This is similar to a simplified LeNet-5.

---

## 🔧 Installation

### 1️⃣ Clone the project
```sh
git clone https://github.com/yourusername/cnn_from_scratch.git
cd cnn_from_scratch
```

### 2️⃣ Install dependencies
```sh
pip install numpy
```

---

## ▶️ Running the Project

### **Train for a few steps:**
```sh
python main.py
```

### **Run test forward pass:**
```sh
python tests/test_sample.py
```

---

## 📊 Example Output

```
Epoch 1 - Loss: 2.302
Epoch 2 - Loss: 2.289
Epoch 3 - Loss: 2.271
```

Sample prediction output:

```
Logits: [[...]]
Probabilities: [[...]]
Predicted class: 2
```

---

## 🤝 Contributing

Pull requests are welcome.  
For major changes, open an issue first to discuss what you'd like to add.

---

## 📝 License

MIT License 



