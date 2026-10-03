# Deep Learning Project Roadmap: Beginner to Master

A project-based path for learning deep learning. Each level builds on the one before it. Do the projects in order, and ship each one before moving on.

---

## How to Use This Guide

**Rules for every project:**

1. **Own repo.** One GitHub repository per project.
2. **README with results.** State what you built, what numbers you got, and how to run it.
3. **One ablation.** Remove or change one thing (e.g. remove dropout) and record what happens. This teaches what each part actually does.
4. **One failure analysis.** Look at what your model gets wrong and write down why.
5. **Time limit.** Decide a deadline before you start. Ship it even if it is ugly.
6. **No blind copying.** If you follow a tutorial, change something: a different dataset, a different architecture, a different hyperparameter.

**Time split:** Spend at most 30% of your time on tutorials and videos. Spend 70% building, breaking, and fixing.

**Tools you will use throughout:**

| Tool | What it is for |
|------|----------------|
| Python | Main language |
| PyTorch | Deep learning library (this guide assumes PyTorch) |
| NumPy | Array math |
| Matplotlib | Plots and image display |
| Jupyter or VS Code | Where you write code |
| Git and GitHub | Saving and sharing your work |
| Google Colab or Kaggle Notebooks | Free GPUs if you have no GPU |

---

## Level 0: Foundations

**Time:** 1 to 2 weeks
**Goal:** Understand how a neural network learns, not just how to call library functions.

### Project 0.1: Autograd Engine From Scratch
- **What you build:** A tiny library that computes gradients automatically (like a mini `loss.backward()`).
- **Skills learned:** Backpropagation, chain rule, computation graphs.
- **Resource:** Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd" (YouTube).
- **Done when:** Your engine trains a small network to classify simple 2D points.

### Project 0.2: MNIST Done Properly
- **What you build:** A digit classifier with a clean training pipeline.
- **Skills learned:**
  - Splitting data into **train / validation / test** sets
  - Writing separate `train()` and `evaluate()` functions
  - Plotting loss curves
  - Viewing misclassified images
- **Dataset:** MNIST (built into `torchvision`).
- **Done when:** Test accuracy is about 96 to 97% with a simple MLP, and you have a loss curve plot.

### Project 0.3: Overfit One Batch
- **What you build:** Take a single small batch (e.g. 32 images) and train until the model memorizes it perfectly.
- **Why:** If a model cannot reach ~100% on one tiny batch, there is a bug in your code. This is the single most useful debugging trick.
- **Done when:** Training loss goes to nearly zero on that one batch.

**Concepts to know by the end of Level 0:** tensors, forward pass, loss function, gradient, backward pass, optimizer step, learning rate, epoch, batch.

---

## Level 1: Beginner

**Time:** 2 to 4 weeks
**Goal:** Build real models on real data, and learn the standard training toolkit.

### Project 1.1: CIFAR-10 CNN From Scratch
- **What you build:** A convolutional neural network (CNN) that classifies 32x32 color images into 10 classes.
- **Dataset:** CIFAR-10 (built into `torchvision`).
- **Skills learned:** Convolution, pooling, BatchNorm, dropout, data augmentation.
- **Method:** Start with a plain CNN. Then add BatchNorm, dropout, and augmentation **one at a time**. Record the accuracy after each addition.
- **Target:** 85% test accuracy or more.
- **Done when:** You have a table showing what each technique gave you.

### Project 1.2: Transfer Learning on Your Own Dataset
- **What you build:** Fine-tune a pretrained model (`resnet18`) to classify images you collected yourself (plants, food, cars, anything with 3+ classes).
- **Skills learned:** Pretrained models, freezing layers, fine-tuning, building a custom `Dataset` class.
- **Tip:** Collect at least 50 to 100 images per class.
- **Done when:** You compare "train from scratch" against "fine-tune" and see the difference.

### Project 1.3: Tabular Data, MLP vs XGBoost
- **What you build:** Predict a number or class from spreadsheet-style data using both a neural network and XGBoost.
- **Dataset:** Any Kaggle tabular dataset (e.g. house prices, Titanic).
- **Skills learned:** Feature preprocessing, fair model comparison.
- **Lesson:** XGBoost often wins on tabular data. Deep learning is not always the right tool.

### Training toolkit to learn in Level 1
- `Dataset` and `DataLoader` classes
- Learning rate schedulers
- Early stopping
- Saving and loading checkpoints
- Weight decay and regularization
- Reading a loss curve (underfitting vs overfitting)

---

## Level 2: Intermediate

**Time:** 1 to 2 months
**Goal:** Cover the main areas of deep learning: language, generation, segmentation, detection.

### Project 2.1: makemore (Character-Level Language Model)
- **What you build:** A model that generates new names (or words) one character at a time. Build it in stages: bigram, then MLP, then RNN, then Transformer.
- **Skills learned:** Language modeling, embeddings, sequence models, next-token prediction.
- **Resource:** Andrej Karpathy, "Neural Networks: Zero to Hero" (makemore videos).

### Project 2.2: Autoencoder and VAE
- **What you build:** An autoencoder that compresses images, then a Variational Autoencoder (VAE) that can generate new ones.
- **Dataset:** Fashion-MNIST (easier) or CelebA (harder).
- **Skills learned:** Latent space, reconstruction loss, KL divergence.
- **Done when:** You can sample new images and walk smoothly between two images in latent space.

### Project 2.3: U-Net Image Segmentation
- **What you build:** A model that labels every pixel in an image (e.g. pet vs background).
- **Dataset:** Oxford-IIIT Pets, or a medical imaging dataset.
- **Skills learned:** Encoder-decoder design, skip connections, IoU and Dice metrics.

### Project 2.4: Object Detection With YOLO
- **What you build:** Fine-tune a YOLO model to find and box objects in images using your own labeled data.
- **Skills learned:** Bounding boxes, IoU, non-max suppression (NMS), mAP.
- **Tools:** Ultralytics YOLO, Roboflow or LabelImg for labeling.

### Project 2.5: Sentiment Classifier (LSTM vs BERT)
- **What you build:** Classify movie or product reviews as positive or negative. Build once with an LSTM, once by fine-tuning BERT, then compare.
- **Dataset:** IMDB reviews.
- **Skills learned:** Tokenization, padding, attention masks, Hugging Face `transformers`.

### Habit to start in Level 2
- **Read one paper per week.** Start with: ResNet, Attention Is All You Need, Adam, Batch Normalization, U-Net.

---

## Level 3: Advanced

**Time:** 2 to 4 months
**Goal:** Build the models behind modern AI from scratch.

### Project 3.1: GPT From Scratch
- **What you build:** A small GPT-style language model, including a tokenizer, attention layers, and training loop. Train it on your own text (e.g. Shakespeare, song lyrics, your own writing).
- **Skills learned:** Self-attention, positional encoding, byte-pair encoding (BPE), text generation, sampling (temperature, top-k).
- **Resources:** Karpathy "Let's build GPT: from scratch, in code, spelled out" and the nanoGPT repo.

### Project 3.2: Vision Transformer (ViT) From Scratch
- **What you build:** A transformer for images, trained on CIFAR-10.
- **Lesson:** ViT usually loses to a CNN on small datasets. Find out why (CNNs have built-in assumptions about images; ViT must learn them from data).

### Project 3.3: Diffusion Model (DDPM)
- **What you build:** A model that generates images by learning to remove noise step by step. Start with MNIST, then CIFAR-10.
- **Skills learned:** Noise schedules, denoising objective, sampling loops.

### Project 3.4: GAN (DCGAN, then WGAN-GP)
- **What you build:** A generator and discriminator that train against each other.
- **Skills learned:** Adversarial training, mode collapse, training instability, why WGAN-GP helps.
- **Note:** GANs are hard to train. Expect failures. Learning from them is the point.

### Project 3.5: Reinforcement Learning
- **What you build:** An agent that learns by trial and error. Start with DQN on CartPole, then PPO on LunarLander (or Atari).
- **Skills learned:** Rewards, policies, replay buffers, exploration vs exploitation.
- **Tools:** Gymnasium, Stable-Baselines3 (for comparison).

### Project 3.6: LoRA / QLoRA Fine-Tuning of an LLM
- **What you build:** Fine-tune an open language model (Llama, Qwen, Mistral) on a narrow task, with a real evaluation set to prove it improved.
- **Skills learned:** Parameter-efficient fine-tuning, quantization, evaluation of language models.
- **Tools:** Hugging Face `transformers`, `peft`, `bitsandbytes`.

---

## Level 4: Master

**Time:** 6+ months (this level never really ends)
**Goal:** Work at research and production level.

### Project 4.1: Reproduce a Paper
- **What you build:** Re-implement a published paper and try to match its reported numbers.
- **Why:** This is hard and teaches more than anything else. When your numbers do not match, finding out why is the lesson.
- **Where to find papers:** arXiv, Papers With Code.

### Project 4.2: Multi-GPU and Efficient Training
- **What you build:** Take an existing training script and make it faster and larger.
- **Skills learned:**
  - Distributed training: DDP, FSDP
  - Mixed precision: `torch.autocast`
  - Gradient accumulation
  - Gradient checkpointing

### Project 4.3: Custom GPU Kernel
- **What you build:** A custom operation written in Triton or CUDA. Profile before and after.
- **Skills learned:** GPU memory, parallelism, finding real bottlenecks with `torch.profiler`.

### Project 4.4: End-to-End ML System
- **What you build:** A full pipeline: data collection, training, evaluation, serving through an API, and monitoring.
- **Skills learned:** FastAPI, Docker, model versioning, latency and cost measurement, experiment tracking (Weights & Biases or MLflow).

### Project 4.5: Original Research
- **What you build:** Pick one small open question. Run proper experiments with ablations and multiple random seeds. Write it up as a blog post or arXiv paper.

### Project 4.6: Open Source Contribution
- **What you do:** Fix a bug, add docs, or add a feature in PyTorch, Hugging Face, or another ML library. Start with issues labeled "good first issue."

---

## Resources

### Courses (free)
| Course | Best for |
|--------|----------|
| **fast.ai: Practical Deep Learning for Coders** | Top-down, code-first start |
| **Karpathy: Neural Networks: Zero to Hero** | Deep understanding, building from scratch |
| **Stanford CS231n** | Computer vision |
| **Stanford CS224n** | Natural language processing |
| **Hugging Face NLP / LLM Course** | Transformers in practice |
| **Hugging Face Deep RL Course** | Reinforcement learning |

### Books and websites
- **Dive into Deep Learning (d2l.ai):** Free interactive book with code
- **Deep Learning by Goodfellow, Bengio, Courville:** Free online reference
- **PyTorch official tutorials (pytorch.org/tutorials):** Official docs and examples
- **Papers With Code:** Papers matched with implementations
- **The Illustrated Transformer (Jay Alammar):** Best visual explanation of transformers

### Datasets
- **Image:** MNIST, Fashion-MNIST, CIFAR-10, Oxford Pets, CelebA
- **Text:** IMDB, Tiny Shakespeare, WikiText
- **Tabular:** Kaggle datasets
- **Everything else:** Hugging Face Datasets, Kaggle, UCI ML Repository

### Math to learn (only when you hit a wall)
1. **Linear algebra:** vectors, matrices, matrix multiplication
2. **Calculus:** derivatives, chain rule
3. **Probability and statistics:** distributions, expectation, Bayes' rule
4. **Optimization basics:** gradient descent

Good math refreshers: 3Blue1Brown "Essence of Linear Algebra" and "Essence of Calculus" (YouTube).

---

## Common Mistakes to Avoid

| Mistake | Fix |
|---------|-----|
| **Tutorial hell:** watching course after course, building nothing | Cap tutorials at 30% of time. Build something after every lesson. |
| **Not measuring results** | Always split data into train/val/test and report numbers. |
| **Tuning on the test set** | Tune on validation data. Touch the test set once, at the end. |
| **Skipping debugging basics** | Overfit one batch first. Check shapes. Check the loss at step 0. |
| **Never finishing** | Set a deadline. Ship an ugly version. Improve it after. |
| **Chasing quantity** | Three polished projects with analysis beat twenty copied notebooks. |
| **Ignoring baselines** | Always compare against a simple baseline (e.g. logistic regression, XGBoost). |

---

## Quick Checklist Per Project

- [ ] Clear goal and deadline written down
- [ ] Data split into train / validation / test
- [ ] Simple baseline built first
- [ ] Training and evaluation in separate functions
- [ ] Loss curves plotted
- [ ] One ablation run and recorded
- [ ] Errors inspected and explained
- [ ] README with results, how to run, and lessons learned
- [ ] Code pushed to GitHub