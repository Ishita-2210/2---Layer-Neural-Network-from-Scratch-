# 2---Layer-Neural-Network-from-Scratch-
This project implements a 2-layer neural network from scratch using only NumPy, without using any high-level deep learning libraries like TensorFlow or PyTorch.
The model is trained to classify handwritten digits from the MNIST dataset.
Project Structure
│
├── neural_network.py      # Core training and model code
├── dataset_loading.py     # Dataset loading and preprocessing (if separated)
├── README.md              # Project documentation
└── requirements.txt       # Python package requirements (if any)

Requirements
Python 3.x
Numpy

Model Architecture
Input Layer: 784 nodes (flattened 28x28 pixel images)
Hidden Layer: Configurable size (default: 10 nodes, recommended: 64 nodes)
Activation (Hidden): ReLU
Output Layer: 10 nodes (one for each digit class)
Activation (Output): Softmax

Potential Extensions
Implement mini-batch training
Add real-time accuracy and loss plotting
Support for deeper networks (3 or more layers)
Add regularization to prevent overfitting
