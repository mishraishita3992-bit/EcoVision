import tensorflow as tf
import numpy as np
import os

from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


# -----------------------------
# Settings
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP"]


# -----------------------------
# Load training dataset
# -----------------------------
print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# -----------------------------
# Load validation dataset
# -----------------------------
print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Load existing model
# -----------------------------
print("\nLoading existing model...")

model = tf.keras.models.load_model(
    "models/plastic_classifier.keras"
)

print("Existing model loaded successfully!")


# -----------------------------
# Display model structure
# -----------------------------
print("\nModel layers:")

for i, layer in enumerate(model.layers):
    print(i, layer.name)


# -----------------------------
# Find MobileNetV2 base model
# -----------------------------
base_model = None

for layer in model.layers:

    if "mobilenet" in layer.name.lower():

        base_model = layer
        break


if base_model is None:

    print("\nERROR: Could not find MobileNetV2.")
    print("Please send me the 'Model layers' output.")
    exit()


print("\nMobileNetV2 found:")
print(base_model.name)


# -----------------------------
# Unfreeze MobileNetV2
# -----------------------------
base_model.trainable = True


# Freeze earlier layers
# Only last 30 layers will train
for layer in base_model.layers[:-30]:

    layer.trainable = False


# Keep BatchNormalization frozen
for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


print("\nFine-tuning configuration:")
print("Earlier MobileNetV2 layers: FROZEN")
print("Last 30 MobileNetV2 layers: TRAINABLE")
print("BatchNormalization layers: FROZEN")


# -----------------------------
# Calculate class weights
# -----------------------------
print("\nCalculating class weights...")

train_labels = []

for images, labels in train_dataset:

    train_labels.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )


train_labels = np.array(train_labels)


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)


class_weights = {
    i: weight
    for i, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name}: "
        f"{class_weights[i]:.2f}"
    )


# -----------------------------
# Compile
# -----------------------------
print("\nCompiling model...")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# -----------------------------
# Early stopping
# -----------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


# -----------------------------
# Fine-tune
# -----------------------------
print("\n==============================")
print("STARTING FINE-TUNING")
print("==============================\n")


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=20,

    class_weight=class_weights,

    callbacks=[early_stopping]
)


# -----------------------------
# Save model
# -----------------------------
os.makedirs(
    "models",
    exist_ok=True
)


model.save(
    "models/plastic_classifier_finetuned.keras"
)


print("\n==============================")
print("FINE-TUNING COMPLETE")
print("==============================")

print(
    "\nNew model saved as:"
)

print(
    "models/plastic_classifier_finetuned.keras"
)