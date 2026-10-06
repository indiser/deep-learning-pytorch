# CIFAR-10 Image Classifier

A CNN built with PyTorch to classify images from the CIFAR-10 dataset into 10 categories.

## Architecture

`NuralNet` — a 2-block convolutional network:

```
Input (3×32×32)
  → Conv2d(3, 12, 5)  → ReLU → MaxPool2d(2,2)
  → Conv2d(12, 24, 5) → ReLU → MaxPool2d(2,2)
  → Flatten
  → Linear(600, 120) → ReLU
  → Linear(120, 84)  → ReLU
  → Linear(84, 10)   → (logits for 10 classes)
```

Classes: `plane, car, bird, cat, deer, dog, frog, horse, ship, truck`

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` (lr=0.001, betas=(0.9, 0.999), eps=1e-7)
- Epochs: 10, Batch size: 128

## Project Structure

```
Cifer/
├── main.py       # Model definition, training, evaluation, and single-image prediction
├── model.pt      # Saved model weights (state_dict)
├── train/        # Auto-downloaded CIFAR-10 training data (50,000 samples)
├── test/         # Auto-downloaded CIFAR-10 test data (10,000 samples)
└── Readme.md
```

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `Pillow`

## Usage

**Evaluate with saved weights (default):**
```bash
python main.py
```

**Retrain the model:**

Set `IS_TRAINED = False` in `main.py`, then run:
```bash
python main.py
```

This will train for 10 epochs, save weights to `model.pt`, print test accuracy, and predict the class of the image at `Image_PATH`.

## Notes

- Automatically uses GPU (`cuda`) if available, otherwise falls back to CPU.
- CIFAR-10 data is auto-downloaded on first run via `torchvision.datasets.CIFAR10`.
- Custom image prediction resizes any image to 32×32 before inference.
