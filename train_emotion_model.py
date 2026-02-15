"""
Train a custom Emotion Recognition model using FER-2013 dataset.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Dataset paths
TRAIN_DIR = './datasets/fer2013/train'
TEST_DIR = './datasets/fer2013/test'
MODEL_SAVE_PATH = 'custom_emotion_model.h5'

# Image parameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
NUM_CLASSES = 7
EPOCHS = 25

# Emotion classes (in order)
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("=" * 60)
print("Emotion Recognition Model Training")
print("=" * 60)
print(f"Train directory: {TRAIN_DIR}")
print(f"Test directory: {TEST_DIR}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of classes: {NUM_CLASSES}")
print("=" * 60)

# Verify dataset directories exist
if not os.path.exists(TRAIN_DIR):
    raise ValueError(f"Training directory not found: {TRAIN_DIR}")
if not os.path.exists(TEST_DIR):
    raise ValueError(f"Test directory not found: {TEST_DIR}")

# Count samples in each class
print("\nCounting samples in each class...")
for emotion in EMOTION_CLASSES:
    train_count = len(os.listdir(os.path.join(TRAIN_DIR, emotion)))
    test_count = len(os.listdir(os.path.join(TEST_DIR, emotion)))
    print(f"  {emotion}: Train={train_count}, Test={test_count}")

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation/test (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
print("\nCreating data generators...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")

# Build CNN Model
print("\n" + "=" * 60)
print("Building CNN Model...")
print("=" * 60)

def build_emotion_model():
    """
    Build a CNN model for emotion recognition.
    Architecture:
    - 4 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPooling -> Dropout)
    - Flatten
    - Dense layer (512 units) + Dropout
    - Output Dense layer (7 units, Softmax)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(48, 48, 1)),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten
        layers.Flatten(),
        
        # Dense layer
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# Create model
model = build_emotion_model()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
print("\nModel Architecture:")
print("=" * 60)
model.summary()
print("=" * 60)

# Callbacks
callbacks = [
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# Training
print("\n" + "=" * 60)
print("Starting Training...")
print("=" * 60)

# Custom callback to print accuracy after each epoch
class AccuracyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print("-" * 60)

accuracy_callback = AccuracyCallback()
callbacks.append(accuracy_callback)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# Print training history summary
print("\nTraining History Summary:")
print("-" * 60)
for i, (train_acc, val_acc) in enumerate(zip(history.history['accuracy'], 
                                             history.history['val_accuracy']), 1):
    print(f"Epoch {i}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print("=" * 60)
