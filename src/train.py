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
        A TensorFlow OneDeviceStrategy targeting GPU if available, else CPU.
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
    """Train an AE or VAE on the Medical MNIST dataset and save the model.

    Args:
        model_type: Either "AE" or "VAE".
        data_dir: Path to the Medical MNIST root directory (contains class subdirs).
        epochs: Number of training epochs.
        batch_size: Samples per batch.
        models_dir: Directory where the trained model file is saved.

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

    print(f"\n--- Training {model_type} ({epochs} epochs, batch_size={batch_size}) ---")
    start = time.time()
    history = model.fit(dataset, epochs=epochs, verbose=1)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s")

    out_path = Path(models_dir) / f"{model_type.lower()}_v1.keras"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if model_type == "AE":
        model.save(str(out_path))
        print(f"Model saved to {out_path}")
    else:
        enc_path = out_path.with_name("vae_encoder_v1.keras")
        dec_path = out_path.with_name("vae_decoder_v1.keras")
        model.encoder.save(str(enc_path))
        model.decoder.save(str(dec_path))
        print(f"VAE encoder saved to {enc_path}")
        print(f"VAE decoder saved to {dec_path}")

    return history, model
