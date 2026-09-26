import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "models/plastic_classifier_finetuned.keras"
TEST_DIR = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# IMPORTANT:
# This must match the class order used during training.
CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP"]


# -----------------------------
# Load trained model
# -----------------------------
print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# -----------------------------
# Load test dataset
# -----------------------------
print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=CLASS_NAMES,
    label_mode="int"
)

print("\nClasses:")
print(test_dataset.class_names)


# -----------------------------
# Make predictions
# -----------------------------
y_true = []
y_pred = []

print("\nTesting model...")

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# -----------------------------
# Test accuracy
# -----------------------------
accuracy = np.mean(y_true == y_pred)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


# -----------------------------
# Classification report
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# -----------------------------
# Confusion matrix
# -----------------------------
print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

print("\n==============================")
print("TESTING COMPLETE")
print("==============================")