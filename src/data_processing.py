"""Data processing utilities for Medical MNIST dataset."""

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
        Tuple of (image, image) — input and target are identical for AE training.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, img


def get_label_from_path(file_path: tf.Tensor) -> tf.Tensor:
    """Extract an integer class label from the parent directory name.

    Args:
        file_path: Path tensor to a JPEG image.

    Returns:
        Integer label tensor (0–5) corresponding to the class subdirectory.
    """
    # Normalize Windows backslashes so splitting on "/" works on all platforms
    normalized = tf.strings.regex_replace(file_path, r"\\", "/")
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
        shuffle: Whether to shuffle file order before batching.

    Returns:
        Batched, prefetched tf.data.Dataset yielding (image, image) pairs.
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
        Batched, prefetched tf.data.Dataset yielding (image, label) pairs.
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
