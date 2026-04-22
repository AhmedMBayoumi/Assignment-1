"""Tests for model module."""

import tensorflow as tf

from src.model import LATENT_DIM, Sampling, VAE, build_ae, build_vae_components


def test_ae_output_shape() -> None:
    """AE output shape matches input shape."""
    ae = build_ae()
    x = tf.random.uniform((2, 64, 64, 1))
    out = ae(x, training=False)
    assert out.shape == (2, 64, 64, 1)


def test_ae_output_range() -> None:
    """AE outputs are in [0, 1] due to sigmoid final activation."""
    ae = build_ae()
    x = tf.random.uniform((2, 64, 64, 1))
    out = ae(x, training=False)
    assert float(tf.reduce_max(out)) <= 1.0 + 1e-5
    assert float(tf.reduce_min(out)) >= -1e-5


def test_vae_encoder_output_shapes() -> None:
    """VAE encoder outputs [z_mean, z_log_var, z] each of shape (B, LATENT_DIM)."""
    encoder, _ = build_vae_components()
    x = tf.random.uniform((2, 64, 64, 1))
    z_mean, z_log_var, z = encoder(x, training=False)
    assert z_mean.shape == (2, LATENT_DIM)
    assert z_log_var.shape == (2, LATENT_DIM)
    assert z.shape == (2, LATENT_DIM)


def test_vae_decoder_output_shape() -> None:
    """VAE decoder maps (B, LATENT_DIM) → (B, 64, 64, 1)."""
    _, decoder = build_vae_components()
    z = tf.random.normal((2, LATENT_DIM))
    out = decoder(z, training=False)
    assert out.shape == (2, 64, 64, 1)


def test_vae_call_output_shape() -> None:
    """VAE.call returns reconstructed images of correct shape."""
    encoder, decoder = build_vae_components()
    vae = VAE(encoder, decoder)
    x = tf.random.uniform((2, 64, 64, 1))
    out = vae(x, training=False)
    assert out.shape == (2, 64, 64, 1)


def test_vae_metric_names() -> None:
    """VAE exposes the three expected loss metric names."""
    encoder, decoder = build_vae_components()
    vae = VAE(encoder, decoder)
    names = [m.name for m in vae.metrics]
    assert "total_loss" in names
    assert "reconstruction_loss" in names
    assert "kl_loss" in names


def test_sampling_output_shape() -> None:
    """Sampling layer output shape matches input shape."""
    layer = Sampling()
    z_mean = tf.zeros((3, LATENT_DIM))
    z_log_var = tf.zeros((3, LATENT_DIM))
    z = layer([z_mean, z_log_var])
    assert z.shape == (3, LATENT_DIM)
