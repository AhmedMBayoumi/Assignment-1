# DSAI 490 — Assignment 1: Representation Learning with Autoencoders

Autoencoder (AE) and Variational Autoencoder (VAE) trained on the Medical MNIST dataset for reconstruction, latent space visualization, sample generation, and denoising.

## Project Structure

```
├── data/
│   ├── raw/medical-mnist/   # JPEG images (6 class subdirectories)
│   └── processed/
├── models/                  # Saved .keras model files after training
├── notebooks/
│   └── experiment.ipynb     # Main experiment notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # tf.data pipeline
│   ├── model.py             # AE, VAE, Sampling layer definitions
│   └── train.py             # Training utilities, GPU/CPU auto-detection
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

Download **Medical MNIST** from Kaggle:  
https://www.kaggle.com/datasets/andrewmvd/medical-mnist

Extract into `data/raw/medical-mnist/` so the structure looks like:

```
data/raw/medical-mnist/
├── AbdomenCT/
├── BreastMRI/
├── ChestCT/
├── CXR/
├── Hand/
└── HeadCT/
```

## Run

### Tests

```bash
pytest tests/ -v
```

### Experiment Notebook

```bash
cd notebooks
jupyter notebook experiment.ipynb
```

Run all cells top-to-bottom. Training takes ~20 epochs per model.

## Models

| Model | Architecture | Loss |
|-------|-------------|------|
| AE | CNN Encoder → Dense(64) → CNN Decoder | MSE |
| VAE | CNN Encoder → μ,σ → Sampling → CNN Decoder | BCE + KL Divergence |

Both models use `LATENT_DIM = 64`. Latent space is visualized with PCA and t-SNE.

## Configuration

Edit the **Configuration** cell at the top of `notebooks/experiment.ipynb`:

```python
DATA_DIR   = '../data/raw/medical-mnist'
MODELS_DIR = '../models'
EPOCHS     = 20
BATCH_SIZE = 64
```

## Requirements

- Python 3.10+
- TensorFlow 2.13+
- See `requirements.txt` for full list
