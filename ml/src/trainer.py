"""
TremorTrace ML Pipeline — Trainer

Two-phase training strategy for MobileNetV2 transfer learning:
  Phase 1: Frozen base — train classification head only (high learning rate)
  Phase 2: Unfreeze top layers — fine-tune with low learning rate

Optimized for small datasets (~72 training images per drawing type).
"""

import os
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

try:
    from src.config import (
        PHASE1_EPOCHS, PHASE1_LR, PHASE1_PATIENCE,
        PHASE2_EPOCHS, PHASE2_LR, PHASE2_PATIENCE,
        UNFREEZE_LAYERS,
    )
except ImportError:
    from config import (
        PHASE1_EPOCHS, PHASE1_LR, PHASE1_PATIENCE,
        PHASE2_EPOCHS, PHASE2_LR, PHASE2_PATIENCE,
        UNFREEZE_LAYERS,
    )


def get_callbacks(name, phase, save_dir):
    """
    Creates training callbacks for a given phase.

    Args:
        name: str, model name (e.g. 'spiral', 'wave')
        phase: int, training phase (1 or 2)
        save_dir: str, directory to save model checkpoints

    Returns:
        list: Keras callback objects
    """
    patience = PHASE1_PATIENCE if phase == 1 else PHASE2_PATIENCE

    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(save_dir, f'{name}_phase{phase}_best.keras'),
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=1,
        ),
    ]

    return callbacks


def train_two_phase(model, base_model, train_gen, val_gen, name, save_dir):
    """
    Trains a model using the two-phase transfer learning strategy.

    Phase 1: All base model layers frozen, train head with high learning rate.
    Phase 2: Unfreeze top N layers of base model, fine-tune with low learning rate.

    Args:
        model: compiled Keras Model
        base_model: the MobileNetV2 base model (for freeze/unfreeze control)
        train_gen: training data generator
        val_gen: validation data generator
        name: str, model identifier (e.g. 'spiral', 'wave')
        save_dir: str, directory to save models and checkpoints

    Returns:
        dict: Merged training history from both phases with keys:
            - Standard Keras history keys (loss, val_loss, accuracy, etc.)
            - 'phase2_start': int, epoch index where Phase 2 began
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"📊 Class weights for {name}: {class_weight_dict}")

    total_start = time.time()

    # =========================================================================
    # Phase 1: Frozen base — train head only
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🔒 Phase 1: Training {name} head (base frozen)")
    print(f"   Learning rate: {PHASE1_LR}")
    print(f"   Max epochs: {PHASE1_EPOCHS}")
    print(f"   Early stopping patience: {PHASE1_PATIENCE}")
    print(f"{'='*60}\n")

    phase1_start = time.time()

    phase1_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(name, 1, save_dir),
        verbose=1,
    )

    phase1_duration = time.time() - phase1_start
    phase1_epochs_run = len(phase1_history.history['loss'])

    p1_val_acc = phase1_history.history['val_accuracy'][-1]
    p1_val_auc = phase1_history.history['val_auc'][-1]
    print(f"\n✅ {name.capitalize()} Phase 1 complete — "
          f"val_accuracy: {p1_val_acc:.1%}, val_auc: {p1_val_auc:.4f} "
          f"(took {phase1_duration:.1f}s, {phase1_epochs_run} epochs)")

    # =========================================================================
    # Phase 2: Unfreeze top layers — fine-tune
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🔓 Phase 2: Fine-tuning {name} (unfreezing last {UNFREEZE_LAYERS} layers)")
    print(f"   Learning rate: {PHASE2_LR}")
    print(f"   Max epochs: {PHASE2_EPOCHS}")
    print(f"   Early stopping patience: {PHASE2_PATIENCE}")
    print(f"{'='*60}\n")

    # Unfreeze last N layers of the base model
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    # Recompile with much lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LR),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
        ],
    )

    phase2_start = time.time()

    phase2_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(name, 2, save_dir),
        verbose=1,
    )

    phase2_duration = time.time() - phase2_start
    phase2_epochs_run = len(phase2_history.history['loss'])

    p2_val_acc = phase2_history.history['val_accuracy'][-1]
    p2_val_auc = phase2_history.history['val_auc'][-1]
    print(f"\n✅ {name.capitalize()} Phase 2 complete — "
          f"val_accuracy: {p2_val_acc:.1%}, val_auc: {p2_val_auc:.4f} "
          f"(took {phase2_duration:.1f}s, {phase2_epochs_run} epochs)")

    # =========================================================================
    # Merge histories and save
    # =========================================================================
    merged_history = {}
    for key in phase1_history.history:
        merged_history[key] = (
            phase1_history.history[key] + phase2_history.history[key]
        )
    merged_history['phase2_start'] = phase1_epochs_run

    # Save final model
    final_path = os.path.join(save_dir, f'{name}_final.keras')
    model.save(final_path)

    total_duration = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🏁 {name.capitalize()} training complete!")
    print(f"   Total time: {total_duration:.1f}s")
    print(f"   Total epochs: {phase1_epochs_run + phase2_epochs_run}")
    print(f"   Final val_accuracy: {p2_val_acc:.1%}")
    print(f"   Final val_auc: {p2_val_auc:.4f}")
    print(f"   Model saved: {final_path}")
    print(f"{'='*60}\n")

    return merged_history
