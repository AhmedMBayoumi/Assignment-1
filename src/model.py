"""AE and VAE model definitions for Medical MNIST."""

from typing import List, Tuple

import tensorflow as tf


LATENT_DIM: int = 64
IMAGE_SHAPE: Tuple[int, int, int] = (64, 64, 1)


def build_ae() -> tf.keras.Model:
    """Build a convolutional Autoencoder.

    Encoder: Conv2D(32, stride=2) → Conv2D(64, stride=2) → Flatten → Dense(LATENT_DIM, relu)
    Decoder: Dense → Reshape(16,16,64) → Conv2DTranspose(64) → Conv2DTranspose(32) → Conv2DTranspose(1, sigmoid)

    Returns:
        Full autoencoder tf.keras.Model mapping (B, 64, 64, 1) → (B, 64, 64, 1).
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
        """Sample a latent vector from the Gaussian defined by (z_mean, z_log_var).

        Args:
            inputs: Tuple of (z_mean, z_log_var), each of shape (B, LATENT_DIM).

        Returns:
            Sampled z of shape (B, LATENT_DIM).
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_components() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Build VAE encoder and decoder as separate Keras models.

    Encoder outputs [z_mean, z_log_var, z] where z is the reparameterized sample.
    Decoder architecture mirrors the AE decoder.

    Returns:
        Tuple of (encoder, decoder) Keras models.
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
    """Variational Autoencoder with combined reconstruction + KL divergence loss.

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
        """Return the list of tracked loss metrics for this model.

        Returns:
            List containing total_loss, reconstruction_loss, and kl_loss trackers.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode inputs to latent space then decode back to image space.

        Args:
            inputs: Image batch of shape (B, 64, 64, 1).
            training: Whether in training mode.

        Returns:
            Reconstructed images of shape (B, 64, 64, 1).
        """
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data: tf.Tensor) -> dict:
        """Compute reconstruction (BCE) + KL divergence loss and apply gradients.

        Args:
            data: Batch from tf.data, may be an (images, images) tuple.

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

    def test_step(self, data: tf.Tensor) -> dict:
        """Evaluate the model without updating weights.

        Args:
            data: Batch from tf.data, may be an (images, images) tuple.

        Returns:
            Dict with keys: loss, reconstruction_loss, kl_loss.
        """
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
