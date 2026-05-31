# Medical Imaging AI Evaluation Pipeline

A reproducible validation pipeline for chest X-ray pathology classification using the NIH ChestX-ray14 dataset. Built with DenseNet-121, Monte Carlo Dropout uncertainty quantification, failure mode classification, and MLflow experiment tracking. Docker-containerized for deployment.

---

## Results

### Model Performance (8,278 NIH ChestX-ray14 images)

| Metric | Value |
|---|---|
| Best Validation AUC | **0.7901** |
| Test Mean AUC | **0.7224** |
| Diagnostic Accuracy | **52.7%** |
| Training Epochs | 18 (early stopping) |
| GPU | NVIDIA GeForce RTX 4050 Laptop |

### Per-Class AUC-ROC (Test Set)

| Pathology | AUC | F1 | Accuracy |
|---|---|---|---|
| Hernia | **0.9706** | 0.0833 | 0.9857 |
| Cardiomegaly | **0.8161** | 0.3989 | 0.8625 |
| Edema | **0.7887** | 0.1244 | 0.6512 |
| Fibrosis | **0.7809** | 0.1438 | 0.8292 |
| Effusion | 0.7480 | 0.3876 | 0.5756 |
| Pneumothorax | 0.7349 | 0.3017 | 0.7647 |
| Emphysema | 0.7191 | 0.1595 | 0.8214 |
| Pleural Thickening | 0.6528 | 0.1423 | 0.7014 |
| Atelectasis | 0.6516 | 0.2684 | 0.5130 |
| Mass | 0.6573 | 0.1689 | 0.7112 |
| Consolidation | 0.6389 | 0.1692 | 0.6030 |
| Nodule | 0.6381 | 0.1474 | 0.6532 |
| Infiltration | 0.6215 | 0.4005 | 0.4087 |
| Pneumonia | 0.5549 | 0.0763 | 0.8579 |

> Low F1 scores are expected due to heavy class imbalance in the NIH dataset (e.g. Hernia < 0.2% prevalence). AUC-ROC is the primary metric for imbalanced multi-label classification.

---

## Architecture

```
Input (224×224 RGB)
    ↓
DenseNet-121 (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
MC Dropout (p=0.3) ← active at inference for uncertainty
    ↓
Linear (1024 → 14)
    ↓
Sigmoid → Multi-label predictions
```

- **Model**: DenseNet-121 (~7M parameters)
- **Loss**: Weighted Binary Cross-Entropy (handles class imbalance)
- **Uncertainty**: Monte Carlo Dropout with 20 stochastic forward passes
- **Failure Modes**: 6 categories (False Positive/Negative × High Confidence/Uncertain + Rare Class Miss + Multi-label Confusion)

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── download.py          # Kaggle API download
│   │   ├── preprocess.py        # CLAHE + resize + train/val/test splits
│   │   └── dataset.py           # PyTorch Dataset + augmentation
│   ├── models/
│   │   ├── densenet.py          # DenseNet-121 with MCDropout
│   │   └── uncertainty.py       # Monte Carlo Dropout inference
│   ├── training/
│   │   ├── trainer.py           # Training loop + MLflow + early stopping
│   │   └── losses.py            # Weighted BCE loss
│   └── evaluation/
│       ├── metrics.py           # AUC-ROC, F1, diagnostic accuracy
│       └── failure_modes.py     # Failure mode classification
├── experiments/configs/
│   └── base_config.yaml         # All hyperparameters
├── train.py                     # Main training entry point
├── evaluate.py                  # Full evaluation pipeline
├── Dockerfile                   # CUDA 12.1 + cuDNN 8
├── docker-compose.yml           # Pipeline + MLflow UI services
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11
- NVIDIA GPU with CUDA 12.1 support
- Kaggle account with API token

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/medical-imaging-ai-eval-pipeline.git
cd medical-imaging-ai-eval-pipeline

python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Kaggle Credentials

Place your `kaggle.json` at `~/.kaggle/kaggle.json`:
```json
{"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}
```

---

## Running the Pipeline

### 1. Download Data (~7.6 GB)
```bash
python -m src.data.download --config experiments/configs/base_config.yaml
```

### 2. Preprocess Images
```bash
python -m src.data.preprocess --config experiments/configs/base_config.yaml
```

### 3. Train
```bash
python train.py
```

### 4. Evaluate
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### 5. MLflow Dashboard
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## Docker

```bash
docker-compose up --build
```

Runs the full pipeline in a CUDA-enabled container with the MLflow UI available at port 5000. Mount your Kaggle credentials before running:

```bash
# Place kaggle.json at ~/.kaggle/kaggle.json before running
docker-compose up --build
```

---

## Configuration

All hyperparameters are in [`experiments/configs/base_config.yaml`](experiments/configs/base_config.yaml):

```yaml
training:
  epochs: 30
  batch_size: 16
  learning_rate: 0.0001
  early_stopping_patience: 7

model:
  architecture: densenet121
  dropout_rate: 0.3

uncertainty:
  mc_dropout_passes: 20
```

---

## Dataset

- **NIH ChestX-ray14** — 112,120 frontal chest X-rays from 30,805 patients
- 14 pathology labels: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
- This implementation uses a 8,278-image subset (`images_001` + `images_002`)
- Full dataset available at [kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## Scaling Up

To reproduce the full 85–89% AUC reported in the CheXNet paper, download all 12 image archives (45 GB total) — zero code changes required:

```yaml
# experiments/configs/base_config.yaml
data:
  archives_to_download:
    - "images_001.zip"
    - "images_002.zip"
    - "images_003.zip"
    # ... through images_012.zip
```

---

## Tech Stack

`PyTorch` · `DenseNet-121` · `MLflow` · `Docker` · `CUDA` · `albumentations` · `scikit-learn` · `pydicom` · `OpenCV`
