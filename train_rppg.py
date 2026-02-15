"""
Training script for rPPG PhysNet 3D-CNN on UBFC-rPPG.
- Checks Apple Metal GPU.
- 80/20 train/val split by subjects.
- ModelCheckpoint, ReduceLROnPlateau, EarlyStopping.
- 20 epochs. workers=1 recommended (MediaPipe + open/close per batch).
- Saves best model as deep_pulse_model.keras.

Robust retrain (--robust): load deep_pulse_model.keras, train 5 more epochs with
25% static (zero-motion) samples (label 0.0) to reduce mean-overfitting. Saves robust_pulse_model.keras.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loader import discover_subject_folders, UBFCDataGenerator
from rppg_model import build_rppg_model, INPUT_SHAPE

DATA_ROOT = "./datasets/ubfc/data_folder"
BATCH_SIZE = 16
EPOCHS = 20
SAVE_BEST = "deep_pulse_model.keras"
VAL_SPLIT = 0.2
RANDOM_STATE = 42

# Robust retrain: load previous model, static ratio, epochs, save path
LOAD_MODEL_ROBUST = "deep_pulse_model.keras"
ROBUST_STATIC_RATIO = 0.25
ROBUST_EPOCHS = 5
ROBUST_SAVE = "robust_pulse_model.keras"


def main(robust: bool = False):
    # --- Channel-last and Metal ---
    keras.backend.set_image_data_format("channels_last")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU(s) available (e.g. Apple Metal): {[g.name for g in gpus]}")
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    else:
        print("No GPU found; using CPU.")

    # --- Robust retrain: epochs, save path, static ratio ---
    epochs = ROBUST_EPOCHS if robust else EPOCHS
    save_best = ROBUST_SAVE if robust else SAVE_BEST
    static_ratio = ROBUST_STATIC_RATIO if robust else 0.0
    if robust:
        print(f"Robust retrain: load {LOAD_MODEL_ROBUST}, {ROBUST_EPOCHS} epochs, {int(100*ROBUST_STATIC_RATIO)}% static (label 0), save {ROBUST_SAVE}")

    # --- Discover and split subjects ---
    root = os.path.abspath(os.path.expanduser(DATA_ROOT))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Data root not found: {root}")

    subjects = discover_subject_folders(root)
    if not subjects:
        raise FileNotFoundError(f"No subject folders (vid.avi + ground_truth.txt) under {root}")

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * VAL_SPLIT))
    val_subjects = subjects[:n_val]
    train_subjects = subjects[n_val:]
    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")

    # --- Generators (open/process/close per batch; no full load) ---
    train_gen = UBFCDataGenerator(
        subject_folders=train_subjects,
        batch_size=BATCH_SIZE,
        clip_len=30,
        face_size=(128, 128),
        shuffle=True,
        normalize="divide",
        use_face_detection=True,
        static_ratio=static_ratio,
    )
    val_gen = UBFCDataGenerator(
        subject_folders=val_subjects,
        batch_size=BATCH_SIZE,
        clip_len=30,
        face_size=(128, 128),
        shuffle=False,
        normalize="divide",
        use_face_detection=True,
        static_ratio=static_ratio,
    )

    n_train = len(train_gen._index)
    n_val_samples = len(val_gen._index)
    print(f"Train chunks: {n_train}, Val chunks: {n_val_samples}")

    if n_train == 0:
        raise RuntimeError("No training chunks. Check videos and ground_truth.txt format.")

    use_val = n_val_samples > 0
    mon = "val_loss" if use_val else "loss"

    # --- Model: load previous for robust retrain, else build from scratch ---
    if robust:
        if not os.path.isfile(LOAD_MODEL_ROBUST):
            raise FileNotFoundError(f"Robust retrain requires {LOAD_MODEL_ROBUST}. Train base model first.")
        model = keras.models.load_model(LOAD_MODEL_ROBUST)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
        print(f"Loaded {LOAD_MODEL_ROBUST} for robust retrain.")
    else:
        model = build_rppg_model(input_shape=INPUT_SHAPE)
    model.summary()

    # --- Callbacks ---
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_best,
            monitor=mon,
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=mon,
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=mon,
            patience=5,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # --- Train ---
    model.fit(
        train_gen,
        validation_data=val_gen if use_val else None,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(save_best)
    print(f"Best model saved to {save_best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train rPPG PhysNet. Use --robust to retrain with static samples.")
    ap.add_argument("--robust", action="store_true", help="Load deep_pulse_model.keras, add 25%% static (label 0), 5 epochs, save robust_pulse_model.keras")
    args = ap.parse_args()
    main(robust=args.robust)
