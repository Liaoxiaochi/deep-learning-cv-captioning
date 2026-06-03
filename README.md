# Deep Learning: Image Classification & Caption Generation

> Coursework for **COMP5625M Deep Learning** (MSc Advanced Computer Science, University of Leeds).
> End-to-end PyTorch implementation covering convolutional image classification and
> image-to-text caption generation with both RNN and Transformer (CLIP + GPT-2) decoders.

This project is split into two parts:

| Part | Task | Techniques |
|------|------|-----------|
| **1** | Image classification on **TinyImageNet30** (30 classes, 64×64) | 5-layer CNN, data augmentation, CutMix, cosine annealing, ensembling + TTA, Kaggle leaderboard |
| **2** | **Image caption generation** | ResNet-50 feature encoder + RNN decoder; CLIP vision encoder + Transformer mapper + GPT-2 decoder |

---

## Part 1 — CNN Image Classification

A custom 5-convolution CNN (`TinyCNN`) trained from scratch on the 30-class TinyImageNet
subset (13,500 images at 64×64), with the predictions submitted to a private Kaggle
competition leaderboard.

**Key components** (see [`src/train_improved.py`](src/train_improved.py)):

- **Architecture** — 5 conv blocks (64→128→256→256→512 channels) with BatchNorm + ReLU +
  MaxPool, followed by a 1024-unit dense head with dropout.
- **Regularisation / augmentation** — random crop, horizontal flip, rotation, colour jitter,
  random grayscale, random erasing, label smoothing, and **CutMix**.
- **Optimisation** — AdamW with weight decay and a **cosine-annealing** learning-rate schedule.
- **Inference** — a **3-seed ensemble** combined with **test-time augmentation (TTA)** (original,
  horizontal flip, centre crop, crop + flip) by averaging logits.

The pipeline addresses the classic overfitting problem on a small dataset through
aggressive augmentation and ensembling.

## Part 2 — Image Caption Generation

Two complementary captioning approaches on the COCO dataset:

1. **CNN encoder + RNN decoder** ([`src/helperDL.py`](src/helperDL.py)) — a pretrained
   **ResNet-50** extracts image feature vectors; captions are cleaned, tokenised, and a
   vocabulary is built (min-frequency filtering); sequences are batched with padding /
   `pack_padded_sequence` for an RNN language model.

2. **CLIP + Transformer + GPT-2** ([`src/model_CLIP.py`](src/model_CLIP.py)) — a frozen
   **CLIP** vision encoder produces image embeddings, a **Transformer encoder** maps them
   into the GPT-2 embedding space, and a partially-frozen **GPT-2** decoder generates the
   caption autoregressively with temperature sampling. Only the mapping network and the
   first/last GPT-2 layers are fine-tuned, keeping the model lightweight.

---

## Repository Structure

```
deep-learning-cv-captioning/
├── notebooks/
│   └── assessment.ipynb         # main coursework notebook (Part 1 + Part 2)
├── src/
│   ├── train_improved.py        # Part 1: improved CNN training, ensemble + TTA, Kaggle submission
│   ├── helperDL.py              # Part 2: dataset/vocab helpers, ResNet-50 encoder
│   └── model_CLIP.py            # Part 2: CLIP encoder + Transformer mapper + GPT-2 decoder
├── docs/
│   └── rendered_notebook.html   # notebook with all cell outputs / figures rendered
├── .gitignore
└── README.md
```

> **Note** — datasets (TinyImageNet30, COCO), trained model weights (`*.pth`), and Kaggle
> submission files are intentionally excluded from version control (see `.gitignore`).
> The rendered HTML in `docs/` preserves the executed results and figures.

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `Hugging Face Transformers` (CLIP, GPT-2) ·
`scikit-learn` · `pandas` · `NumPy` · `Pillow` · `Matplotlib`

## How to Run

The notebook is designed for **Google Colab with a GPU runtime**:

1. Open `notebooks/assessment.ipynb` in Colab and select a GPU runtime.
2. Provide the TinyImageNet30 dataset (`train_set/`, `test_set/`) for Part 1 and the COCO
   captions/images for Part 2.
3. Run the cells top to bottom. For Part 1, `src/train_improved.py` can be run as a cell to
   train the ensemble and generate the Kaggle submission CSV.

---

*Author: Xiaochi Liao (sc21xl2) — University of Leeds.*
