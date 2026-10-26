# Two-layer neural network from scratch

An educational NumPy implementation of handwritten-digit classification: **784 inputs → 10 ReLU units → 10 softmax outputs**, trained with explicit forward propagation, backpropagation and gradient descent.

## Files and data

- `Neural_network_scratch.ipynb`: the digit classifier and development-set evaluation.
- `Capstone.ipynb`: a separate historical parking-pricing experiment, preserved here. It contains Colab/Pathway-specific work and requires its own data; it is not part of the classifier setup.

The repository contains notebooks, not `neural_network.py` or `dataset_loading.py`.

Provide the labeled MNIST/Digit Recognizer training CSV (for example from Kaggle's Digit Recognizer competition) as `data/train.csv`, or set `MNIST_CSV` to its path. Obtain it under the source's terms. The first column must be `label` (digits 0–9), followed by 784 numeric pixels in [0,255], with more than 1,000 complete rows. Data is not redistributed here.

## Run

Python 3.11+:

```sh
python -m venv .venv
# Activate .venv before installing.
python -m pip install -r requirements.txt
jupyter lab Neural_network_scratch.ipynb
```

Run cells in order. Seed 42 makes the shuffle and initialization repeatable. The first 1,000 shuffled examples form a development set; the rest train the model for 500 iterations. Both splits use pixel scaling by 255. Training prints accuracy every ten iterations; the final cell evaluates the held-out development split. There is no independently validated benchmark claim. Saved outputs are historical; rerun the notebook for current results.

## Test the math without downloading MNIST

```sh
python -m pip install pytest
python -m pytest -q
```

Tests check stable softmax, gradients against finite differences and one-hot labels when a batch omits digit 9. The hidden layer is fixed at ten units in this implementation. Full training requires the external CSV and is separate from CI's small synthetic checks.
