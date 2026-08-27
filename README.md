# 🛰️ Deep Learning Exam: Satellite Cloud Cover Classification

> **Intermediate–Advanced Level** | Python · PyTorch · Remote Sensing

---

## Overview

You are given a real-world satellite imagery task: **classify cloud cover in multi-spectral satellite images**. Your goal is to build, train, validate, and test a deep learning model from scratch using PyTorch.

This exam tests your ability to:
- Source and prepare real geospatial/satellite data
- Design and train a deep learning pipeline end-to-end
- Evaluate model performance rigorously
- Write clean, well-documented, reproducible code

> ✅ **AI Tool Policy:** You are permitted to use generative AI tools (e.g., ChatGPT, Copilot) **only for data exploration and analysis** — for example, to help you interpret spectral bands, understand dataset statistics, or debug preprocessing logic. Using AI to write your core model architecture, training loop, or evaluation code is discouraged and will be apparent during review. Your code should reflect *your* understanding.

---

## The Problem

Satellites capture multi-spectral images of Earth's surface. A persistent challenge in remote sensing is **cloud cover detection** — clouds obstruct the surface and must be identified accurately before downstream analysis (e.g., crop monitoring, disaster response, urban mapping).

Your task is to build a **binary semantic segmentation model** that, given a multi-spectral satellite image tile, produces a **pixel-wise cloud mask** (cloud = 1, clear = 0).

---

## Dataset

You will use the **[38-Cloud Dataset](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)** — a benchmark dataset of Landsat-8 satellite imagery patches with labeled cloud masks.

### Step 1 — Clone the dataset repository

```bash
git clone https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset.git
```

The dataset contains:
- **Red**, **Green**, **Blue**, and **Near-Infrared (NIR)** spectral band patches (`.TIF` format, 384×384 px each)
- Corresponding binary **ground-truth cloud masks**
- A CSV file listing training and test patches

> 💡 **Tip:** Use ChatGPT or another AI tool to help you understand what each spectral band represents and how NIR differs from visible bands. Understanding the data domain will help you make better preprocessing choices.

### Step 2 — Install required dependencies

You will need `rasterio` for reading `.TIF` geospatial files. Install it from source or via pip:

```bash
pip install rasterio
```

For additional geospatial utilities, optionally clone `torchgeo`:

```bash
git clone https://github.com/microsoft/torchgeo.git
cd torchgeo
pip install -e .
```

---

## Tasks

Complete **all** of the following tasks. Each is graded independently.

---

### Task 1 — Data Loading & Exploration (20 pts)

Implement a custom PyTorch `Dataset` class that:

- Reads the 4-band `.TIF` image patches (R, G, B, NIR) using `rasterio`
- Loads the corresponding binary mask
- Applies appropriate normalization per band
- Supports a configurable train/validation split (suggested: 80/20)

**Deliverables:**
- `dataset.py` — your `Dataset` and `DataLoader` setup
- A short **exploratory data analysis (EDA)** section in a Jupyter Notebook or script that shows:
  - Sample images visualized in RGB and NIR
  - Distribution of cloud vs. non-cloud pixels in the training set
  - Any class imbalance observations and how you plan to address them

> 💡 You may use ChatGPT to help you interpret what the band statistics mean physically (e.g., why clouds appear bright in NIR), but implement the loading logic yourself.

---

### Task 2 — Model Architecture (25 pts)

Design and implement a deep learning segmentation model in PyTorch. You are free to choose any architecture, but your choice must be **justified in a short written explanation** (3–5 sentences).

**Requirements:**
- The model must accept **4-channel input** (R, G, B, NIR)
- Output a single-channel **probability map** (sigmoid output, same spatial resolution as input)
- The architecture must be implemented **manually** — do not simply load a pretrained segmentation model with default weights and call `.fit()`

**Suggested starting point (not required):** A U-Net style encoder-decoder. A plain CNN is also acceptable if well-designed.

**Deliverables:**
- `model.py` — your model class with clear comments explaining each component
- A `README` section or docstring justifying your architecture choice

---

### Task 3 — Training Pipeline (25 pts)

Implement a complete training loop that includes:

- **Loss function:** Choose an appropriate loss for binary segmentation (e.g., BCE, Dice Loss, or a combination). Justify your choice.
- **Optimizer:** Adam or AdamW with a reasonable learning rate schedule
- **Metrics tracked per epoch:** Training loss, Validation loss, IoU (Intersection over Union), and F1-score
- **Early stopping:** Stop training if validation loss does not improve for `N` consecutive epochs (make `N` configurable)
- **Checkpointing:** Save the best model weights based on validation IoU

**Deliverables:**
- `train.py` — your full training script
- A **training curve plot** (loss and IoU vs. epoch) saved as `outputs/training_curves.png`

---

### Task 4 — Evaluation & Testing (20 pts)

Evaluate your trained model on the held-out **test set** provided by the dataset.

Report the following metrics:
| Metric | Description |
|--------|-------------|
| Pixel Accuracy | % of correctly classified pixels |
| IoU (Jaccard) | Intersection over Union for cloud class |
| F1 / Dice Score | Harmonic mean of precision and recall |
| Precision | True positive rate for cloud pixels |
| Recall | Sensitivity for cloud pixels |

Additionally, provide **at least 5 qualitative visualizations** showing: the input RGB image, the ground-truth mask, and your predicted mask side-by-side. Save them to `outputs/predictions/`.

**Deliverables:**
- `evaluate.py` — evaluation script
- `outputs/metrics.json` — a JSON file with all reported metrics
- `outputs/predictions/` — sample prediction visualizations

---

### Task 5 — Reflection & Analysis (10 pts)

In a file called `REPORT.md`, write a concise technical report (500–800 words) addressing:

1. **Data challenges:** What preprocessing decisions did you make and why? Did class imbalance affect your approach?
2. **Architecture rationale:** Why did you choose your model design? What are its limitations?
3. **Results analysis:** Where does your model succeed and fail? Look at your qualitative predictions — are there patterns in the errors?
4. **Improvements:** If given more time, what would you change? (e.g., data augmentation strategies, architecture changes, multi-scale inputs)

---

## Submission Structure

Your final submission should be a GitHub repository structured as follows:

```
your-repo/
├── README.md              ← this file
├── REPORT.md              ← Task 5 report
├── dataset.py             ← Task 1
├── model.py               ← Task 2
├── train.py               ← Task 3
├── evaluate.py            ← Task 4
├── requirements.txt       ← all dependencies with versions
├── notebooks/
│   └── eda.ipynb          ← exploratory data analysis
└── outputs/
    ├── training_curves.png
    ├── metrics.json
    └── predictions/
        └── *.png
```

> **Note:** Do **not** commit the dataset itself. Your `README.md` or a `setup.sh` script should include instructions to download/clone it.

---

## Grading Rubric

| Task | Points | Key Criteria |
|------|--------|-------------|
| Task 1 — Data Loading & EDA | 20 | Correct multi-band loading, normalization, EDA quality |
| Task 2 — Model Architecture | 25 | Design soundness, 4-channel support, code clarity, justification |
| Task 3 — Training Pipeline | 25 | Loss choice, metrics tracking, early stopping, checkpointing |
| Task 4 — Evaluation | 20 | Metric correctness, qualitative analysis, test set usage |
| Task 5 — Report | 10 | Analytical depth, honesty about limitations |
| **Total** | **100** | |

**Bonus (up to 10 pts):**
- Implement data augmentation (random flips, rotations, color jitter on RGB bands only) — +5 pts
- Experiment with at least 2 loss functions and compare results — +5 pts

---

## Rules & Constraints

- ✅ Use Python 3.8+ and PyTorch 2.x
- ✅ You may use standard libraries: `numpy`, `matplotlib`, `scikit-learn`, `rasterio`, `albumentations`
- ✅ AI tools are allowed for **data analysis and interpretation only**
- ❌ Do not use high-level wrappers that abstract away the training loop (e.g., PyTorch Lightning's `trainer.fit()` without understanding what it does)
- ❌ Do not use pretrained segmentation models with frozen weights as your final submission
- ❌ Do not hardcode file paths — use relative paths or config variables

---

## Tips

- Read the [38-Cloud paper](https://arxiv.org/abs/1901.10077) for domain context. It is short and informative.
- NIR is highly informative for cloud detection — don't drop it!
- Clouds are often the **majority class** — think carefully about your loss function.
- Commit your code incrementally. A clean git history demonstrates good engineering habits.

---

*Good luck. We look forward to seeing your approach.*
