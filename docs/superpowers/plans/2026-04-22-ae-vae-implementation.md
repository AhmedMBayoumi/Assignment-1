# AE/VAE Medical MNIST Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, runnable AE/VAE project for Medical MNIST with modular src/, tested code, and an experiment notebook.

**Architecture:** Modular src package (data_processing, model, train) imported by experiment notebook. CNN-based encoder-decoder for both AE (MSE loss) and VAE (BCE + KL loss). tf.data pipeline for efficient local loading.

**Tech Stack:** Python 3.10+, TensorFlow 2.x, NumPy, Matplotlib, scikit-learn (t-SNE/PCA), pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `models/.gitkeep`, `notebooks/.gitkeep`
- Create: `src/__init__.py`, `tests/__init__.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p data/raw data/processed models notebooks src tests
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep
```

- [ ] **Step 2: Write requirements.txt**

```
tensorflow>=2.13.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scipy>=1.11.0
jupyter>=1.0.0
ipykernel>=6.0.0
pytest>=7.4.0
```

- [ ] **Step 3: Create virtual environment and install**

```bash
python -m venv venv
venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

- [ ] **Step 4: Write src/__init__.py**

```python
"""Source package for AE/VAE Medical MNIST project."""
```

- [ ] **Step 5: Write tests/__init__.py**

Empty file.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat: project scaffolding and requirements"
```

---

### Task 2: Download Medical MNIST Dataset

**Files:**
- Populate: `data/raw/medical-mnist/` with JPEG subdirectories

- [ ] **Step 1: Download via Kaggle CLI**

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/ then:
kaggle datasets download -d andrewmvd/medical-mnist -p data/raw/
```

- [ ] **Step 2: Unzip**

```bash
cd data/raw
unzip medical-mnist.zip -d medical-mnist/
```

Expected structure:
```
data/raw/medical-mnist/
  AbdomenCT/   (*.jpeg)
  BreastMRI/   (*.jpeg)
  ChestCT/     (*.jpeg)
  CXR/         (*.jpeg)
  Hand/        (*.jpeg)
  HeadCT/      (*.jpeg)
```

---

### Task 3: src/data_processing.py

**Files:**
- Create: `src/data_processing.py`
- Test: `tests/test_data_processing.py`

- [ ] **Step 1: Write tests/test_data_processing.py**

```python
"""Tests for data_processing module."""

from pathlib import Path

import pytest
import tensorflow as tf

from src.data_processing import get_dataset, get_labeled_dataset, process_path

IMAGE_SIZE = (64, 64, 1)


def _make_jpeg(path: Path) -> None:
    dummy = tf.zeros((28, 28, 1), dtype=tf.uint8)
    jpeg_bytes = tf.image.encode_jpeg(dummy).numpy()
    path.write_bytes(jpeg_bytes)


def test_process_path_shape_and_dtype(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpeg"
    _make_jpeg(img_path)
    img, target = process_path(str(img_path))
    assert img.shape == IMAGE_SIZE
    assert img.dtype == tf.float32
    assert target.shape == IMAGE_SIZE


def test_process_path_normalized(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpeg"
    dummy = tf.fill((28, 28, 1), 255)
    dummy = tf.cast(dummy, tf.uint8)
    (tmp_path / "img.jpeg").write_bytes(tf.image.encode_jpeg(dummy).numpy())
    img, _ = process_path(str(img_path))
    assert float(tf.reduce_max(img)) <= 1.0
    assert float(tf.reduce_min(img)) >= 0.0


def test_get_dataset_batch_shape(tmp_path: Path) -> None:
    cls_dir = tmp_path / "ClassA"
    cls_dir.mkdir()
    for i in range(10):
        _make_jpeg(cls_dir / f"img_{i}.jpeg")
    ds = get_dataset(str(tmp_path), batch_size=4)
    for images, targets in ds.take(1):
        assert images.shape == (4, 64, 64, 1)
        assert targets.shape == (4, 64, 64, 1)
        assert images.dtype == tf.float32


def test_get_labeled_dataset_shape(tmp_path: Path) -> None:
    cls_dir = tmp_path / "ClassA"
    cls_dir.mkdir()
    for i in range(6):
        _make_jpeg(cls_dir / f"img_{i}.jpeg")
    ds = get_labeled_dataset(str(tmp_path), batch_size=4)
    for images, labels in ds.take(1):
        assert images.shape == (4, 64, 64, 1)
        assert labels.shape == (4,)
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_data_processing.py -v
```
Expected: ImportError (module not yet written)

- [ ] **Step 3: Write src/data_processing.py**

```python
"""Data processing utilities for Medical MNIST dataset."""

import os
from pathlib import Path
from typing import Tuple

import tensorflow as tf


IMAGE_SIZE: Tuple[int, int] = (64, 64)
CHANNELS: int = 1

CLASS_NAMES = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]


def process_path(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read, decode, normalize, and resize a JPEG image.

    Args:
        file_path: Path to a JPEG image file.

    Returns:
        Tuple of (image, image) for autoencoder training (input == target).
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, img


def get_label_from_path(file_path: tf.Tensor) -> tf.Tensor:
    """Extract integer class label from the parent directory name.

    Args:
        file_path: Path tensor to a JPEG image.

    Returns:
        Integer label tensor (0–5).
    """
    # Normalize Windows backslashes to forward slashes
    normalized = tf.strings.regex_replace(file_path, r"\\\\", "/")
    parts = tf.strings.split(normalized, "/")
    label_str = parts[-2]
    label_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=CLASS_NAMES,
            values=list(range(len(CLASS_NAMES))),
        ),
        default_value=0,
    )
    return label_table.lookup(label_str)


def get_dataset(
    data_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from Medical MNIST JPEG files.

    Args:
        data_dir: Root directory containing one subdirectory per class.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle file order.

    Returns:
        Batched, prefetched Dataset yielding (image, image) pairs.
    """
    file_pattern = str(Path(data_dir) / "*" / "*.jpeg")
    list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_labeled_dataset(
    data_dir: str,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """Build a labeled tf.data pipeline for latent space visualization.

    Args:
        data_dir: Root directory containing one subdirectory per class.
        batch_size: Number of samples per batch.

    Returns:
        Batched, prefetched Dataset yielding (image, label) pairs.
    """
    file_pattern = str(Path(data_dir) / "*" / "*.jpeg")
    list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)

    def _process_with_label(fp: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img, _ = process_path(fp)
        label = get_label_from_path(fp)
        return img, label

    ds = list_ds.map(_process_with_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_data_processing.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_processing.py tests/test_data_processing.py
git commit -m "feat: tf.data pipeline for Medical MNIST"
```

---

### Task 4: src/model.py

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write tests/test_model.py**

```python
"""Tests for model module."""

import tensorflow as tf

from src.model import LATENT_DIM, Sampling, VAE, build_ae, build_vae_components


def test_ae_output_shape() -> None:
    ae = build_ae()
    x = tf.random.uniform((2, 64, 64, 1))
    out = ae(x, training=False)
    assert out.shape == (2, 64, 64, 1)


def test_ae_output_range() -> None:
    ae = build_ae()
    x = tf.random.uniform((2, 64, 64, 1))
    out = ae(x, training=False)
    assert float(tf.reduce_max(out)) <= 1.0 + 1e-5
    assert float(tf.reduce_min(out)) >= -1e-5


def test_vae_encoder_output_shapes() -> None:
    encoder, _ = build_vae_components()
    x = tf.random.uniform((2, 64, 64, 1))
    z_mean, z_log_var, z = encoder(x, training=False)
    assert z_mean.shape == (2, LATENT_DIM)
    assert z_log_var.shape == (2, LATENT_DIM)
    assert z.shape == (2, LATENT_DIM)


def test_vae_decoder_output_shape() -> None:
    _, decoder = build_vae_components()
    z = tf.random.normal((2, LATENT_DIM))
    out = decoder(z, training=False)
    assert out.shape == (2, 64, 64, 1)


def test_vae_call_output_shape() -> None:
    encoder, decoder = build_vae_components()
    vae = VAE(encoder, decoder)
    x = tf.random.uniform((2, 64, 64, 1))
    out = vae(x, training=False)
    assert out.shape == (2, 64, 64, 1)


def test_vae_metric_names() -> None:
    encoder, decoder = build_vae_components()
    vae = VAE(encoder, decoder)
    names = [m.name for m in vae.metrics]
    assert "total_loss" in names
    assert "reconstruction_loss" in names
    assert "kl_loss" in names


def test_sampling_output_shape() -> None:
    layer = Sampling()
    z_mean = tf.zeros((3, LATENT_DIM))
    z_log_var = tf.zeros((3, LATENT_DIM))
    z = layer([z_mean, z_log_var])
    assert z.shape == (3, LATENT_DIM)
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_model.py -v
```
Expected: ImportError

- [ ] **Step 3: Write src/model.py**

```python
"""AE and VAE model definitions for Medical MNIST."""

from typing import List, Tuple

import tensorflow as tf


LATENT_DIM: int = 64
IMAGE_SHAPE: Tuple[int, int, int] = (64, 64, 1)


def build_ae() -> tf.keras.Model:
    """Build a convolutional Autoencoder.

    Encoder: Conv2D(32) → Conv2D(64) → Flatten → Dense(LATENT_DIM, relu)
    Decoder: Dense → Reshape(16,16,64) → Conv2DTranspose x2 → Conv2DTranspose(sigmoid)

    Returns:
        Full autoencoder tf.keras.Model with input/output shape (B, 64, 64, 1).
    """
    encoder_inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(LATENT_DIM, activation="relu")(x)
    encoder = tf.keras.Model(encoder_inputs, latent, name="ae_encoder")

    latent_inputs = tf.keras.Input(shape=(LATENT_DIM,))
    x = tf.keras.layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((16, 16, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="ae_decoder")

    return tf.keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)), name="autoencoder")


class Sampling(tf.keras.layers.Layer):
    """Reparameterization trick layer: z = μ + σ·ε, ε ~ N(0, I)."""

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Sample latent vector from Gaussian defined by (z_mean, z_log_var).

        Args:
            inputs: Tuple of (z_mean, z_log_var), each shape (B, LATENT_DIM).

        Returns:
            Sampled z of shape (B, LATENT_DIM).
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_components() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Build VAE encoder and decoder as separate Keras models.

    Encoder outputs [z_mean, z_log_var, z].
    Decoder maps a latent vector back to image space.

    Returns:
        Tuple of (encoder, decoder).
    """
    encoder_inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="vae_encoder")

    latent_inputs = tf.keras.Input(shape=(LATENT_DIM,))
    x = tf.keras.layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((16, 16, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="vae_decoder")

    return encoder, decoder


class VAE(tf.keras.Model):
    """Variational Autoencoder with combined BCE + KL divergence loss.

    Args:
        encoder: Keras model outputting [z_mean, z_log_var, z].
        decoder: Keras model mapping (B, LATENT_DIM) → (B, 64, 64, 1).
    """

    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode then decode inputs.

        Args:
            inputs: Image batch of shape (B, 64, 64, 1).
            training: Whether in training mode.

        Returns:
            Reconstructed images of shape (B, 64, 64, 1).
        """
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data: tf.Tensor) -> dict:
        """Compute reconstruction + KL loss and apply gradients.

        Args:
            data: Batch from tf.data, may be (images, images) tuple.

        Returns:
            Dict with keys: loss, reconstruction_loss, kl_loss.
        """
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_model.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: AE and VAE model definitions"
```

---

### Task 5: src/train.py

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write src/train.py**

```python
"""Training utilities for AE and VAE on Medical MNIST."""

import time
from pathlib import Path
from typing import Literal, Tuple

import tensorflow as tf

from src.data_processing import get_dataset
from src.model import VAE, build_ae, build_vae_components


ModelType = Literal["AE", "VAE"]


def get_strategy() -> tf.distribute.Strategy:
    """Return the best available distribution strategy (GPU → CPU fallback).

    Returns:
        A TensorFlow distribution strategy.
    """
    gpus = tf.config.list_physical_devices("GPU")
    device = "/gpu:0" if gpus else "/cpu:0"
    print(f"Using device: {device}")
    return tf.distribute.OneDeviceStrategy(device=device)


def run_training(
    model_type: ModelType,
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    models_dir: str = "models",
) -> Tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """Train an AE or VAE and save the resulting model.

    Args:
        model_type: "AE" or "VAE".
        data_dir: Path to Medical MNIST root directory.
        epochs: Number of training epochs.
        batch_size: Samples per batch.
        models_dir: Directory to save the trained model file.

    Returns:
        Tuple of (training history, trained model).
    """
    tf.keras.backend.clear_session()
    strategy = get_strategy()
    dataset = get_dataset(data_dir, batch_size=batch_size)

    with strategy.scope():
        if model_type == "AE":
            model = build_ae()
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        else:
            encoder, decoder = build_vae_components()
            model = VAE(encoder, decoder)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    print(f"\n--- Training {model_type} ({epochs} epochs, batch={batch_size}) ---")
    start = time.time()
    history = model.fit(dataset, epochs=epochs, verbose=1)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s")

    out_path = Path(models_dir) / f"{model_type.lower()}_v1.keras"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved to {out_path}")

    return history, model
```

- [ ] **Step 2: Commit**

```bash
git add src/train.py
git commit -m "feat: training utilities with GPU/CPU auto-detection"
```

---

### Task 6: notebooks/experiment.ipynb

**Files:**
- Create: `notebooks/experiment.ipynb`

Full notebook with 8 sections. Each section is a separate cell group.

- [ ] **Step 1: Create notebook file** (see implementation — full JSON written to `notebooks/experiment.ipynb`)

- [ ] **Step 2: Verify notebook runs cell-by-cell without error**

```bash
cd notebooks
jupyter nbconvert --to notebook --execute experiment.ipynb --output experiment_executed.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/experiment.ipynb
git commit -m "feat: experiment notebook with reconstruction, latent viz, generation, denoising"
```

---

### Task 7: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md** (see implementation)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and usage instructions"
```
