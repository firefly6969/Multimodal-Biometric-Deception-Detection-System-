"""
PhysNet-inspired 3D-CNN for rPPG heart rate regression.
Input: (30, 128, 128, 3) [Time, Height, Width, RGB], channel-last for Mac (Metal).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_T = 30
INPUT_H = 128
INPUT_W = 128
INPUT_C = 3
INPUT_SHAPE = (INPUT_T, INPUT_H, INPUT_W, INPUT_C)


def build_rppg_model(input_shape: tuple = INPUT_SHAPE) -> keras.Model:
    """
    PhysNet 3D-CNN: Conv3D(16)->tanh->AveragePooling3D, Conv3D(32)->tanh->AveragePooling3D,
    Flatten, Dense(1). Compiled with Adam 1e-4, MSE.
    """
    keras.backend.set_image_data_format("channels_last")

    inp = layers.Input(shape=input_shape, dtype=tf.float32, name="video_input")

    # Conv3D(16, 3x3x3) -> Tanh -> AveragePooling3D
    x = layers.Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation="tanh", name="conv3d_1")(inp)
    x = layers.AveragePooling3D(pool_size=(2, 2, 2), padding="same", name="avgpool_1")(x)

    # Conv3D(32, 3x3x3) -> Tanh -> AveragePooling3D
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="tanh", name="conv3d_2")(x)
    x = layers.AveragePooling3D(pool_size=(2, 2, 2), padding="same", name="avgpool_2")(x)

    x = layers.Flatten(name="flatten")(x)
    out = layers.Dense(1, activation="linear", name="hr_bpm")(x)

    model = keras.Model(inputs=inp, outputs=out, name="rppg_physnet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model
