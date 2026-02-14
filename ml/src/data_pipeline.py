"""
TremorTrace ML Pipeline — Data Pipeline

Dataset loading and augmentation for spiral and wave drawing classification.
Drawing-type-specific augmentation strategies to preserve tremor signal integrity.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from src.config import IMG_SIZE, BATCH_SIZE, RANDOM_SEED
except ImportError:
    from config import IMG_SIZE, BATCH_SIZE, RANDOM_SEED


def get_augmentation_config(drawing_type):
    """
    Returns ImageDataGenerator keyword arguments tailored to the drawing type.

    Spiral drawings are rotationally invariant, so they get full 360° rotation.
    Wave drawings have orientation constraints, so rotation is limited to ±15°
    with horizontal flip allowed.

    NEVER uses shear_range (distorts tremor signal).
    NEVER uses vertical_flip for waves (changes meaning).

    Args:
        drawing_type: str, either 'spiral' or 'wave'

    Returns:
        dict: Keyword arguments for ImageDataGenerator
    """
    if drawing_type not in ('spiral', 'wave'):
        raise ValueError(f"drawing_type must be 'spiral' or 'wave', got '{drawing_type}'")

    # Shared augmentation parameters
    common_config = {
        'rescale': 1.0 / 255.0,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.15,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest',
    }

    if drawing_type == 'spiral':
        # Spirals are rotationally invariant — any rotation is valid
        common_config['rotation_range'] = 360
    else:
        # Waves have orientation — limit rotation, allow horizontal flip
        common_config['rotation_range'] = 15
        common_config['horizontal_flip'] = True

    return common_config


def create_generators(drawing_type, train_path, test_path):
    """
    Creates training, validation, and test data generators for a drawing type.

    Training data is split 80/20 into train/validation using validation_split.
    Test data uses no augmentation (rescale only).

    Args:
        drawing_type: str, 'spiral' or 'wave'
        train_path: str, path to training directory (with healthy/ and parkinson/ subdirs)
        test_path: str, path to testing directory (with healthy/ and parkinson/ subdirs)

    Returns:
        tuple: (train_gen, val_gen, test_gen) — Keras DirectoryIterator objects
    """
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Training path not found: {train_path}")
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"Testing path not found: {test_path}")

    # Augmented generator for training (with validation split)
    aug_config = get_augmentation_config(drawing_type)
    aug_config['validation_split'] = 0.2
    train_datagen = ImageDataGenerator(**aug_config)

    # Common flow_from_directory parameters
    flow_params = {
        'target_size': (IMG_SIZE, IMG_SIZE),
        'batch_size': BATCH_SIZE,
        'class_mode': 'binary',
        'color_mode': 'rgb',
        'seed': RANDOM_SEED,
    }

    # Training generator (80% of training data)
    train_gen = train_datagen.flow_from_directory(
        train_path,
        subset='training',
        shuffle=True,
        **flow_params,
    )

    # Validation generator (20% of training data)
    val_gen = train_datagen.flow_from_directory(
        train_path,
        subset='validation',
        shuffle=False,
        **flow_params,
    )

    # Test generator — no augmentation, rescale only
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        test_path,
        shuffle=False,
        **flow_params,
    )

    print(f"✅ {drawing_type.capitalize()} generators created:")
    print(f"   Training samples:   {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples:       {test_gen.samples}")
    print(f"   Class indices:      {train_gen.class_indices}")

    return train_gen, val_gen, test_gen
