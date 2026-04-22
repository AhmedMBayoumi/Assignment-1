# AE/VAE Representation Learning — Design Spec
**Date:** 2026-04-22
**Dataset:** Medical MNIST (Kaggle) — grayscale JPEG images, 6 classes

---

## Project Structure

```
├── data/
│   ├── raw/medical-mnist/   # user downloads here (subdirs per class)
│   └── processed/
├── models/                  # saved .keras model files
├── notebooks/
│   └── experiment.ipynb     # main experiment notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   └── train.py
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

---

## Components

### src/data_processing.py
- `process_path(file_path)` — reads JPEG, normalizes to [0,1], resizes to 64x64, returns (img, img)
- `get_dataset(data_dir, batch_size, shuffle=True)` — lists files from `data_dir/*/*.jpeg`, maps process_path, batches, prefetches
- `get_label_from_path(file_path)` — extracts class label from parent dir name (for latent space coloring)
- `get_labeled_dataset(data_dir, batch_size)` — returns (img, label) pairs for visualization only

### src/model.py
- `LATENT_DIM = 64`
- `IMAGE_SIZE = (64, 64, 1)`
- `build_ae()` — CNN encoder→64-dim bottleneck→CNN decoder, returns full `tf.keras.Model`
- `Sampling(tf.keras.layers.Layer)` — reparameterization trick: z = μ + σ·ε
- `build_vae_components()` — returns (encoder, decoder); encoder outputs [z_mean, z_log_var, z]
- `VAE(tf.keras.Model)` — wraps encoder+decoder, custom `train_step` with reconstruction loss (BCE) + KL divergence

### src/train.py
- `get_strategy()` — tries GPU, falls back to CPU via `OneDeviceStrategy`
- `run_training(model_type, data_dir, epochs, batch_size)` — builds model inside strategy scope, fits, saves to `models/{model_type}_v1.keras`, returns (history, model)

### notebooks/experiment.ipynb
Sections (in order):
1. Setup & imports
2. Data loading + sample grid visualization
3. Train AE (20 epochs) + loss curve
4. Train VAE (20 epochs) + loss curves (total, reconstruction, KL)
5. Reconstruction comparison: original vs AE vs VAE (5 images)
6. Latent space visualization: t-SNE and PCA of 64-dim encodings, colored by class
7. VAE sample generation: sample z ~ N(0,I), decode, display grid
8. Denoising robustness: add Gaussian noise (σ=0.2), reconstruct with AE and VAE, compare

### tests/
- `test_data_processing.py` — asserts dataset output shape (B,64,64,1), dtype float32, values in [0,1]
- `test_model.py` — asserts AE output shape, VAE output shape, VAE metrics present

---

## Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Latent dim | 64 | Good reconstruction quality; reduce to 2D with t-SNE/PCA for viz |
| Visualization | t-SNE + PCA | Both give complementary views of latent structure |
| Denoising | Robustness demo | Reuses trained models; satisfies requirement without extra training |
| GPU handling | Auto-detect, CPU fallback | Windows GPU support uncertain |
| Epochs | 20 | Balanced quality vs runtime |
| Batch size | 64 | Standard for this image size |
| Loss (AE) | MSE | Standard for pixel reconstruction |
| Loss (VAE) | BCE + KL | Matches example code; BCE per pixel sum |
