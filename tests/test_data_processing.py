"""Tests for data_processing module."""

from pathlib import Path

import tensorflow as tf

from src.data_processing import (
    CLASS_NAMES,
    get_dataset,
    get_label_from_path,
    get_labeled_dataset,
    process_path,
)

IMAGE_SHAPE = (64, 64, 1)


def _make_jpeg(path: Path) -> None:
    """Write a minimal grayscale JPEG to path."""
    dummy = tf.zeros((28, 28, 1), dtype=tf.uint8)
    path.write_bytes(tf.image.encode_jpeg(dummy).numpy())


def test_process_path_shape_and_dtype(tmp_path: Path) -> None:
    """process_path returns float32 tensors of shape (64, 64, 1)."""
    img_path = tmp_path / "img.jpeg"
    _make_jpeg(img_path)
    img, target = process_path(str(img_path))
    assert img.shape == IMAGE_SHAPE
    assert img.dtype == tf.float32
    assert target.shape == IMAGE_SHAPE


def test_process_path_normalized(tmp_path: Path) -> None:
    """process_path normalizes pixel values to [0, 1]."""
    img_path = tmp_path / "img.jpeg"
    white = tf.cast(tf.fill((28, 28, 1), 255), tf.uint8)
    img_path.write_bytes(tf.image.encode_jpeg(white).numpy())
    img, _ = process_path(str(img_path))
    assert float(tf.reduce_max(img)) <= 1.0
    assert float(tf.reduce_min(img)) >= 0.0


def test_get_dataset_batch_shape(tmp_path: Path) -> None:
    """get_dataset batches images into (B, 64, 64, 1) float32 tensors."""
    cls_dir = tmp_path / "ClassA"
    cls_dir.mkdir()
    for i in range(10):
        _make_jpeg(cls_dir / f"img_{i}.jpeg")
    ds = get_dataset(str(tmp_path), batch_size=4)
    for images, targets in ds.take(1):
        assert images.shape == (4, 64, 64, 1)
        assert targets.shape == (4, 64, 64, 1)
        assert images.dtype == tf.float32


def test_get_dataset_input_equals_target(tmp_path: Path) -> None:
    """get_dataset yields identical images as input and target."""
    cls_dir = tmp_path / "ClassA"
    cls_dir.mkdir()
    for i in range(6):
        _make_jpeg(cls_dir / f"img_{i}.jpeg")
    ds = get_dataset(str(tmp_path), batch_size=4)
    for images, targets in ds.take(1):
        assert tf.reduce_all(tf.equal(images, targets))


def test_get_label_from_path_known_class(tmp_path: Path) -> None:
    """get_label_from_path returns the correct index for a known class directory."""
    img_path = tmp_path / "ChestCT" / "img.jpeg"
    img_path.parent.mkdir()
    _make_jpeg(img_path)
    label = get_label_from_path(str(img_path))
    assert int(label) == CLASS_NAMES.index("ChestCT")


def test_get_label_from_path_unknown_class(tmp_path: Path) -> None:
    """get_label_from_path returns -1 for an unrecognised directory name."""
    img_path = tmp_path / "UnknownClass" / "img.jpeg"
    img_path.parent.mkdir()
    _make_jpeg(img_path)
    label = get_label_from_path(str(img_path))
    assert int(label) == -1


def test_get_labeled_dataset_shape(tmp_path: Path) -> None:
    """get_labeled_dataset yields (image, label) batches of correct shape."""
    cls_dir = tmp_path / "ClassA"
    cls_dir.mkdir()
    for i in range(6):
        _make_jpeg(cls_dir / f"img_{i}.jpeg")
    ds = get_labeled_dataset(str(tmp_path), batch_size=4)
    for images, labels in ds.take(1):
        assert images.shape == (4, 64, 64, 1)
        assert labels.shape == (4,)
