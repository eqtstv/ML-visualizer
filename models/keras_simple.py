import sys
import tensorflow as tf

from tensorflow import keras

sys.path.insert(0, "..")
from tfutils import LossAndAccToCsvCallback

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0


def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


model = get_model()

model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=[LossAndAccToCsvCallback()],
    validation_split=0.2,
)
