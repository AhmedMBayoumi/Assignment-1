"""Data processing utilities for Medical MNIST dataset."""

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf


KAGGLE_DATASET_ID: str = "andrewmvd/medical-mnist"


IMAGE_SIZE: Tuple[int, int] = (64, 64)
CHANNELS: int = 1
CLASS_NAMES: List[str] = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]


def ensure_dataset(data_dir: str) -> str:
    """Ensure the Medical MNIST dataset is available, downloading it if needed.

    Checks whether ``data_dir`` already contains the expected class subdirectories.
    If the data is missing, downloads the dataset via ``kagglehub`` and returns
    the path provided by the download cache.  The original ``data_dir`` is
    returned unchanged when the data is already present.

    Args:
        data_dir: Preferred local path to the Medical MNIST root directory.

    Returns:
        Path to the directory that contains the class subdirectories.
    """
    data_path = Path(data_dir)
    already_present = data_path.exists() and any(
        (data_path / cls).is_dir() for cls in CLASS_NAMES
    )

    if already_present:
        print(f"Dataset found at {data_path}")
        return str(data_path)

    print(f"Dataset not found at {data_path}. Downloading via kagglehub...")
    import kagglehub 

    downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print(f"Dataset downloaded to: {downloaded_path}")
    return downloaded_path


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
    normalized = tf.strings.regex_replace(file_path, r"\\", "/")
    parts = tf.strings.split(normalized, "/")
    label_str = parts[-2]
    label_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=CLASS_NAMES,
            values=tf.cast(list(range(len(CLASS_NAMES))), tf.int32),
        ),
        default_value=-1,
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
    ds = ds.filter(lambda _img, lbl: lbl >= 0)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
