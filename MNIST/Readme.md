# MNIST Handwritten Digit Classifier

A feedforward neural network built with PyTorch to classify handwritten digits from the MNIST dataset.

## Architecture

`ClassificationModel` — a 3-layer fully connected network:

```
Input (28×28 = 784)
  → Linear(784, 32) → ReLU
  → Linear(32, 32)  → ReLU
  → Linear(32, 10)  → (logits for digits 0–9)
```

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` (lr=0.001, betas=(0.9, 0.999), eps=1e-7)
- Epochs: 10, Batch size: 32

## Project Structure

```
MNIST/
├── main.py       # Model definition, training, evaluation, and prediction visualization
├── model.pt      # Saved model weights (state_dict)
├── train/        # Auto-downloaded MNIST training data (60,000 samples)
├── test/         # Auto-downloaded MNIST test data (10,000 samples)
└── Readme.md
```

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `matplotlib`

## Usage

**Evaluate with saved weights (default):**
```bash
python main.py
```

**Retrain the model:**

Set `TRAIN = True` in `main.py`, then run:
```bash
python main.py
```

This will train for 10 epochs, save weights to `model.pt`, print test accuracy, and display a grid of 12 sample predictions (green = correct, red = incorrect).

## Notes

- Automatically uses GPU (`cuda`) if available, otherwise falls back to CPU.
- MNIST data is auto-downloaded on first run via `torchvision.datasets.MNIST`.
