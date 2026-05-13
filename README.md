# Generating and Evaluating Multimodal Misinformation Using Synthetic Misinformers on the MOCHEG Dataset

**BTP Project — Kandula Nagendra (22CS01026) | IIT Bhubaneswar | Supervisor: Dr. Shreya Ghosh**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setting Up the MOCHEG Dataset](#3-setting-up-the-mocheg-dataset)
4. [Environment Setup](#4-environment-setup)
5. [Step 1 — Generate the Synthetic Dataset](#5-step-1--generate-the-synthetic-dataset)
6. [Step 2 — Evaluate with Qwen2.5-VL](#6-step-2--evaluate-with-qwen25-vl)
7. [Output Files Reference](#7-output-files-reference)
8. [Hardware Requirements](#8-hardware-requirements)
9. [Troubleshooting](#9-troubleshooting)
10. [Citation](#10-citation)

---

## 1. Project Overview

This project implements the **Synthetic Misinformer** paradigm from Papadopoulos et al. (MAD '23) on the **MOCHEG** fact-checking dataset.

The pipeline has two stages:

```
MOCHEG corpus  ──►  generate_dataset_clip.py  ──►  mocheg_synthetic_12k.csv
                                                             │
                                                             ▼
                                               evaluate_qwen.py
                                                             │
                                                             ▼
                                               results.csv + summary.json
```

**Stage 1 — Dataset Generation** creates four types of (image, claim, evidence) samples:

| Label               | Technique               | What changes                                              |
| ------------------- | ----------------------- | --------------------------------------------------------- |
| `Truthful`          | Original MOCHEG         | Nothing — ground truth baseline                           |
| `Misleading-OOC`    | CSt-alt (CLIP)          | Image is swapped with hard-negative from similar claim    |
| `Misleading-NEI`    | CLIP-NESt-alt           | Named entity in claim text is replaced from similar donor |
| `Misleading-Hybrid` | CSt-alt + CLIP-NESt-alt | Both image swapped AND entity replaced                    |

**Stage 2 — Evaluation** prompts **Qwen2.5-VL-7B-Instruct** zero-shot to classify each sample as `Supported` or `Refuted` and reports accuracy, F1, confusion matrix, and per-technique breakdown.

---

## 2. Repository Structure

```
project/
├── generate_dataset_clip.py   # Stage 1: generate synthetic dataset
├── evaluate_qwen.py           # Stage 2: evaluate with Qwen2.5-VL
├── README.md                  # This file
├── DATASET_DESCRIPTION.md     # Detailed description of output CSV columns
│
├── mocheg/                    # ← Place MOCHEG data here (see Section 3)
│   ├── train/
│   │   ├── Corpus2.csv
│   │   ├── images/
│   │   ├── img_evidence_qrels.csv
│   │   └── text_evidence_qrels_article_level.csv
│   ├── val/
│   └── test/
│
└── outputs/
    ├── mocheg_synthetic_12k.csv    # Generated dataset (Stage 1 output)
    ├── clip_cache.pkl              # CLIP embedding cache (auto-created)
    ├── results_vision.csv          # Evaluation predictions (Stage 2)
    └── results_vision_summary.json # Summary metrics (Stage 2)
```

---

## 3. Setting Up the MOCHEG Dataset

### 3.1 Download

The MOCHEG dataset is publicly available at:

```
https://doi.org/10.5281/zenodo.6653771
```

Or via the GitHub repository:

```
https://github.com/VT-NLP/Mocheg
```

Download and extract the zip. You will get a folder structure like:

```
MOCHEG/
├── train/
│   ├── Corpus2.csv                         ← claims + evidence + labels
│   ├── images/                             ← proof images
│   ├── img_evidence_qrels.csv              ← image ↔ claim mapping
│   └── text_evidence_qrels_article_level.csv
├── val/
│   ├── Corpus2.csv
│   ├── images/
│   └── img_evidence_qrels.csv
├── test/
│   ├── Corpus2.csv
│   ├── images/
│   └── img_evidence_qrels.csv
└── Corpus3.csv                             ← full text corpus (not required here)
```

### 3.2 Files Used by This Project

| File                           | Required for | Description                                      |
| ------------------------------ | ------------ | ------------------------------------------------ |
| `train/Corpus2.csv`            | Stage 1      | Claims, evidence paragraphs, truthfulness labels |
| `train/img_evidence_qrels.csv` | Stage 1      | Maps each claim to its proof image filename      |
| `train/images/`                | Stage 1 & 2  | Proof image files (JPG/PNG)                      |
| `test/Corpus2.csv`             | Optional     | For evaluating on original MOCHEG test split     |

### 3.3 Corpus2.csv Key Columns

| Column                 | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `claim_id`             | Unique integer ID for the claim                          |
| `Claim`                | The claim text to fact-check                             |
| `cleaned_truthfulness` | `Supported`, `Refuted`, or `NEI`                         |
| `Evidence`             | One paragraph of text evidence (multiple rows per claim) |
| `ruling_outline`       | Short explanation of the verdict                         |
| `Snopes URL`           | Source fact-checking article                             |

### 3.4 img_evidence_qrels.csv Key Columns

| Column        | Description                                        |
| ------------- | -------------------------------------------------- |
| `TOPIC`       | The `claim_id` this image belongs to               |
| `DOCUMENT#`   | Image filename in the collection                   |
| `RELEVANCY`   | `1` = ground-truth proof image, `0` = negative     |
| `evidence_id` | Actual proof image filename (used by this project) |

---

## 4. Environment Setup

### 4.1 Python Version

Python **3.9 or later** is required.

### 4.2 Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 4.3 Install requirements

```bash
pip install -r requirements.txt
```

### 4.4 Install Stage 1 Dependencies (Dataset Generation)

```bash
# Core scientific stack
pip install numpy scikit-learn pillow

# CLIP (OpenAI)
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git

# spaCy + transformer-based NER model (best accuracy)
pip install spacy
python -m spacy download en_core_web_trf

# Alternatively, use the lighter model (faster, slightly less accurate):
# python -m spacy download en_core_web_lg
```

> **GPU note:** CLIP embedding is significantly faster on GPU. If you have CUDA, install the matching `torch+cu*` wheel from https://pytorch.org/get-started/locally/

### 4.5 Install Stage 2 Dependencies (Qwen Evaluation)

```bash
pip install "transformers>=4.49.0" accelerate
pip install pandas scikit-learn tqdm
pip install qwen-vl-utils

# Optional — for 4-bit / 8-bit quantisation to fit in less VRAM:
pip install bitsandbytes
```

### 4.6 Verify Installation

```bash
# Check CLIP
python -c "import clip; print('CLIP ok:', clip.available_models())"

# Check spaCy
python -c "import spacy; nlp = spacy.load('en_core_web_trf'); print('spaCy ok')"

# Check transformers version
python -c "import transformers; print(transformers.__version__)"
# Must be >= 4.49.0 for Qwen2.5-VL support

# Check qwen-vl-utils
python -c "from qwen_vl_utils import process_vision_info; print('qwen-vl-utils ok')"
```

---

## 5. Step 1 — Generate the Synthetic Dataset

### 5.1 Basic Command

```bash
python generate_dataset_clip.py \
    --corpus2     mocheg/train/Corpus2.csv \
    --img_qrels   mocheg/train/img_evidence_qrels.csv \
    --images_dir  mocheg/train/images \
    --output      outputs/mocheg_synthetic_12k.csv \
    --target      3000 \
    --spacy_model en_core_web_trf \
    --clip_model  ViT-L/14 \
    --top_k       10 \
    --embed_cache outputs/clip_cache.pkl
```

or

### Simple Command for Generating Dataset

```bash
chmod +x run.sh
./run.sh
```

This generates **12,000 rows** (3,000 × 4 types).

### 5.2 All Arguments

| Argument        | Required | Default           | Description                                                                                         |
| --------------- | -------- | ----------------- | --------------------------------------------------------------------------------------------------- |
| `--corpus2`     | ✓        | —                 | Path to `train/Corpus2.csv`                                                                         |
| `--img_qrels`   | ✓        | —                 | Path to `train/img_evidence_qrels.csv`                                                              |
| `--images_dir`  | ✓        | —                 | Directory containing proof image files                                                              |
| `--output`      | ✓        | —                 | Output CSV path                                                                                     |
| `--target`      |          | `3000`            | Number of samples per type (total = target × 4)                                                     |
| `--spacy_model` |          | `en_core_web_trf` | spaCy NER model name                                                                                |
| `--clip_model`  |          | `ViT-L/14`        | CLIP model variant (`ViT-B/32` or `ViT-L/14`)                                                       |
| `--top_k`       |          | `10`              | Number of CLIP nearest neighbours for hard-negative selection                                       |
| `--embed_cache` |          | _(empty)_         | Path to save/load CLIP embeddings pickle — **strongly recommended** to avoid re-embedding on reruns |
| `--seed`        |          | `42`              | Random seed for reproducibility                                                                     |

### 5.3 Expected Runtime

| Hardware            | ~8k valid claims | ~11k valid claims |
| ------------------- | ---------------- | ----------------- |
| CPU only (fallback) | ~30 min          | ~60 min           |
| GPU (T4 / RTX 3080) | ~8 min           | ~15 min           |
| GPU (A100)          | ~4 min           | ~8 min            |

The first run embeds all claims and images. Subsequent reruns with `--embed_cache` skip embedding entirely (~1 min total).

### 5.4 Progress Output

```
============================================================
 MOCHEG CLIP-Guided Synthetic Misinformer Generator
============================================================

[1/7] Loading CLIP ...
      CLIP 'ViT-L/14' loaded on cuda.
[2/7] Loading spaCy (en_core_web_trf) ...
      spaCy model 'en_core_web_trf' loaded.

[3/7] Loading Corpus2.csv ...
      11210 unique claims.
[3/7] Loading img_evidence_qrels.csv ...

[4/7] Resolving proof images ...
      Valid claims (claim + proof image + evidence): 8743

[5/7] Building CLIP embedding index ...
      Embedding 8743 claim texts with CLIP ...
      Embedding 8743 proof images with CLIP ...
      text_matrix shape : (8743, 768)
      img_matrix  shape : (8743, 768)
      Pre-computing top-10 image-image neighbors ...
      Pre-computing top-10 text-text neighbors ...

[6/7] Building entity cache ...
      Extracting spaCy/regex entities from 8743 claims ...
      Pool sizes: PERSON=5821, LOC=4102, DATE=6834, ORG=2187, EVENT=341

[7/7] Generating 3000 × 4 samples ...
      Progress → {'Truthful': 1000, 'OOC': 1000, 'NEI': 1000, 'Hybrid': 1000}
      Progress → {'Truthful': 2000, 'OOC': 2000, 'NEI': 2000, 'Hybrid': 2000}
      Progress → {'Truthful': 3000, 'OOC': 3000, 'NEI': 3000, 'Hybrid': 3000}

      Final counts : {'Truthful': 3000, 'OOC': 3000, 'NEI': 3000, 'Hybrid': 3000}
      Total rows   : 12000

✓  Dataset saved → outputs/mocheg_synthetic_12k.csv
   Truthful                        3000
   Misleading-Hybrid               3000
   Misleading-NEI                  3000
   Misleading-OOC                  3000

   NEI entity types swapped:
   PERSON            3618
   LOC               2271
   DATE              1448
   ORG                662
```

---

## 6. Step 2 — Evaluate with Qwen2.5-VL

### Simple Command for running all cases

```bash
chmod +x eval.sh
./eval.sh
```

### 6.1 Vision Mode (Recommended)

```bash
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --images_dir mocheg/train/images \
    --mode       vision \
    --model      Qwen/Qwen2.5-VL-7B-Instruct \
    --no_nei
    --output     outputs/results_vision.csv
```

### 6.2 Text-Only Mode

```bash
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --images_dir mocheg/train/images \
    --mode       text \
    --model      Qwen/Qwen2.5-7B-Instruct \
    --no_nei
    --output     outputs/results_text.csv
```

### 6.3 Vison-Only Mode With No Evidence

```bash
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --images_dir mocheg/train/images \
    --mode       vison \
    --model      Qwen/Qwen2.5-7B-Instruct \
    --no_nei
    --no_evidence
    --output     outputs/results_vison_no_evidence.csv
```

### 6.4 Text-Only Mode With No Evidence

```bash
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --images_dir mocheg/train/images \
    --mode       text \
    --model      Qwen/Qwen2.5-7B-Instruct \
    --no_nei
    --no_evidence
    --output     outputs/results_text_no_evidence.csv
```

### 6.5 Quick Smoke Test (50 rows, 4-bit)

```bash
python evaluate_qwen.py \
    --csv        outputs/mocheg_synthetic_12k.csv \
    --images_dir mocheg/train/images \
    --mode       vision \
    --model      Qwen/Qwen2.5-VL-7B-Instruct \
    --output     outputs/results_test50.csv \
    --max_samples 50 \
    --quantize   4bit
```

### 6.4 All Evaluation Arguments

| Argument           | Default                       | Description                                       |
| ------------------ | ----------------------------- | ------------------------------------------------- |
| `--csv`            | _(required)_                  | Input CSV (must have `new_label` column)          |
| `--images_dir`     | _(required)_                  | Directory containing proof images                 |
| `--mode`           | `vision`                      | `vision` (image + text) or `text` (text only)     |
| `--model`          | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID                              |
| `--output`         | `qwen_eval_results.csv`       | Path for prediction CSV output                    |
| `--max_samples`    | `0` (all)                     | Limit rows for quick testing                      |
| `--quantize`       | `none`                        | `4bit` / `8bit` / `none` — reduces VRAM usage     |
| `--max_new_tokens` | `32`                          | Max tokens generated per prediction               |
| `--device`         | `auto`                        | Device map — `auto`, `cuda:0`, `cpu`              |
| `--no_evidence`    | `False`                       | Omit text evidence from prompt (claim-only)       |
| `--no_nei`         | `False`                       | Drop NEI rows; run 2-class Supported/Refuted only |

### 6.5 Expected Evaluation Output

```
================================================================
  OVERALL EVALUATION [2-class: no NEI]
================================================================
  Total samples        : 9000
  Unknown predictions  : 48  (0.5%)
  Valid predictions    : 8952
  Accuracy (valid)     : 43.7%

  Classification Report (2-class):
               precision    recall  f1-score   support
    supported       0.51      0.68      0.58      4500
      refuted       0.41      0.26      0.32      4452

  Confusion Matrix:
                  supported       refuted
      supported        3106          1394
        refuted        3234          1218

----------------------------------------------------------------
  PER-TECHNIQUE BREAKDOWN
  Technique                        N    Acc  Hit rates per class
----------------------------------------------------------------
  Truthful                      3000   52.1%  [supported=62%, refuted=42%]
  Misleading-OOC                3000   38.4%  [supported=71%, refuted=16%]
  Misleading-NEI                3000   44.2%  [supported=65%, refuted=24%]
  Misleading-Hybrid             3000   41.2%  [supported=68%, refuted=19%]
================================================================
```

---

## 7. Output Files Reference

### 7.1 `mocheg_synthetic_12k.csv` (Stage 1 Output)

See **[DATASET_DESCRIPTION.md](DATASET_DESCRIPTION.md)** for a full column-by-column description.

### 7.2 `results_vision.csv` (Stage 2 Output)

| Column                       | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| `claim_id`                   | MOCHEG claim ID                                                |
| `misinformation_label`       | Truthful / Misleading-OOC / Misleading-NEI / Misleading-Hybrid |
| `target_label`               | Ground-truth label (`supported` / `refuted`)                   |
| `generated_misleading_claim` | The claim text shown to the model                              |
| `image_path`                 | Image file shown to the model                                  |
| `prediction`                 | Model's prediction (`supported` / `refuted` / `unknown`)       |
| `raw_response`               | Model's raw text output before parsing                         |
| `nei_swapped_entity_type`    | Entity type swapped (NEI/Hybrid rows)                          |
| `nei_swapped_from`           | Original entity value                                          |
| `nei_swapped_to`             | Replacement entity value                                       |
| `ooc_donor_claim_id`         | Source claim ID for the swapped image                          |

### 7.3 `results_vision_summary.json` (Stage 2 Output)

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "mode": "vision",
  "no_nei": true,
  "total_samples": 9000,
  "unknown_predictions": 48,
  "accuracy_overall": 43.7,
  "per_label_counts": { "Truthful": 3000, "Misleading-OOC": 3000, ... },
  "target_label_dist": { "supported": 4500, "refuted": 4500 },
  "prediction_dist": { "supported": 6340, "refuted": 2660, "unknown": 48 }
}
```

---

## 8. Hardware Requirements

### Stage 1 — Dataset Generation

| Component | Minimum          | Recommended               |
| --------- | ---------------- | ------------------------- |
| RAM       | 16 GB            | 32 GB                     |
| GPU VRAM  | 0 (CPU fallback) | 8 GB+ (for CLIP ViT-L/14) |
| Disk      | 5 GB free        | 20 GB (images + cache)    |

### Stage 2 — Qwen2.5-VL Evaluation

| Configuration                          | VRAM Required |
| -------------------------------------- | ------------- |
| Full precision (fp32)                  | ~28 GB        |
| Half precision (fp16, default)         | ~16 GB        |
| 8-bit quantisation                     | ~10 GB        |
| 4-bit quantisation (`--quantize 4bit`) | ~6 GB         |

A single **A100 (40 GB)** or **RTX 4090 (24 GB)** handles full fp16 without quantisation. For consumer GPUs (8–12 GB VRAM), use `--quantize 4bit`.

---

## 9. Troubleshooting

### `"Image features and image tokens do not match"` error

Your `transformers` version is too old. Run:

```bash
pip install -U "transformers>=4.49.0"
```

### `ImportError: No module named 'qwen_vl_utils'`

```bash
pip install qwen-vl-utils
```

### `CLIP model not found`

```bash
pip install git+https://github.com/openai/CLIP.git
```

Do **not** use `pip install clip` — that installs a different package.

### spaCy model download fails (no internet on compute node)

Download on a machine with internet and copy:

```bash
# On internet machine:
python -m spacy download en_core_web_trf
python -c "import spacy; print(spacy.util.get_package_path('en_core_web_trf'))"
# Copy the printed path to your compute node
```

### Very few valid claims (< 500)

Check that:

1. `--images_dir` points to `train/images/` containing `.jpg`/`.png` files
2. `--img_qrels` points to `train/img_evidence_qrels.csv` (not the test split's)
3. `--corpus2` points to `train/Corpus2.csv`

```bash
ls mocheg/train/images/ | head -5
wc -l mocheg/train/Corpus2.csv
wc -l mocheg/train/img_evidence_qrels.csv
```

### Out-of-memory during CLIP embedding

Reduce the batch size by editing `clip_embed_texts(batch=256)` and `clip_embed_images(batch=64)` in `generate_dataset_clip.py` to smaller values (e.g. 64 and 16).

### Slow generation without GPU

Use `--clip_model ViT-B/32` (smaller, faster) instead of `ViT-L/14`. Results will be slightly lower quality but still valid for experimentation.

---

## 10. Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{papadopoulos2023synthetic,
  title     = {Synthetic Misinformers: Generating and Combating Multimodal Misinformation},
  author    = {Papadopoulos, Stefanos-Iordanis and Koutlis, Christos and
               Papadopoulos, Symeon and Petrantonakis, Panagiotis C.},
  booktitle = {Proc. 2nd ACM Int. Workshop on Multimedia AI against Disinformation (MAD '23)},
  year      = {2023}
}

@inproceedings{yao2023mocheg,
  title     = {End-to-End Multimodal Fact-Checking and Explanation Generation:
               A Challenging Dataset and Models},
  author    = {Yao, Barry Menglong and Shah, Aditya and Sun, Lichao and
               Cho, Jin-Hee and Huang, Lifu},
  booktitle = {Proc. 46th Int. ACM SIGIR Conference (SIGIR '23)},
  year      = {2023}
}
```
#   G e n 
 
 
