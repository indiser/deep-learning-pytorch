# Deep Learning Projects

A collection of deep learning projects built with PyTorch, progressing from fundamentals to more advanced architectures.

## Projects

| # | Project | Description | Status |
|---|---------|-------------|--------|
| 1 | [MNIST](./MNIST/) | Handwritten digit classification with a fully connected network | ✅ Done |
| 2 | [CIFAR-10](./Cifer/) | Image classification with a CNN (10 classes) | ✅ Done |
| 3 | _Coming soon_ | | 🔜 |

## Setup

All projects share a common set of dependencies:

```bash
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `torchvision`, `matplotlib`, `Pillow`

## Structure

```
DeepLearning/
├── requirements.txt
├── MNIST/              # Project 1 — Digit classifier (MLP)
├── Cifer/              # Project 2 — Image classifier (CNN)
└── <future projects>/
```

Each project is self-contained with its own `main.py`, `model.pt`, and `Readme.md`.
