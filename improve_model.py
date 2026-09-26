import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "models/plastic_classifier.keras"
OUTPUT_PATH = "models/plastic_classifier_v2.keras"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

CLASS_NAMES = ["HDPE", "LDPE", "PET", "PP"]

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading datasets...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    shuffle=False
)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
labels = np.concatenate([
    y.numpy() for _, y in train_dataset
])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=labels
)

class_weights = {
    i: float(weights[i])
    for i in range(len(CLASS_NAMES))
}

print("\nClass weights:")
for i, name in enumerate(CLASS_NAMES):
    print(name, round(class_weights[i], 2))

# -----------------------------
# LOAD EXISTING MODEL
# -----------------------------
print("\nLoading existing model...")

model = tf.keras.models.load_model(MODEL_PATH)

# MobileNetV2 is the third layer
base_model = model.layers[2]

# -----------------------------
# FINE-TUNING
# -----------------------------
print("\nPreparing MobileNetV2 for fine-tuning...")

base_model.trainable = True

# Freeze early layers.
# Train only the later layers, starting at block 13.
for layer in base_model.layers:
    if layer.name.startswith("block_13") or \
       layer.name.startswith("block_14") or \
       layer.name.startswith("block_15") or \
       layer.name.startswith("block_16") or \
       layer.name in ["Conv_1", "out_relu"]:
        layer.trainable = True
    else:
        layer.trainable = False

    # Keep BatchNormalization frozen
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("Trainable MobileNetV2 layers:")
for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------
os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# -----------------------------
# TRAIN
# -----------------------------
print("\nStarting fine-tuning...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
model.save(OUTPUT_PATH)

print("\n==============================")
print("FINE-TUNING COMPLETE")
print("==============================")
print("Saved model:", OUTPUT_PATH)