import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DIR = "dataset/train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP"]

MODEL_PATHS = {
    "Original Model": "models/plastic_classifier.keras",
    "Fine-tuned Model": "models/plastic_classifier_finetuned.keras"
}


print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
print(train_dataset.class_names)

print("\nNumber of training images:", 333)


for model_name, model_path in MODEL_PATHS.items():

    print("\n")
    print("=" * 50)
    print(model_name)
    print("=" * 50)

    print("\nLoading model...")

    model = tf.keras.models.load_model(model_path)

    print("Model loaded successfully!")

    y_true = []
    y_pred = []

    print("\nTesting training images...")

    for images, labels in train_dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(
        y_true == y_pred
    )

    print("\nTraining Accuracy:")
    print(
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(cm)

    print("\nCorrect predictions:")

    correct = np.sum(
        y_true == y_pred
    )

    print(
        f"{correct} / {len(y_true)}"
    )

    print("\nIncorrect predictions:")

    incorrect = np.sum(
        y_true != y_pred
    )

    print(
        f"{incorrect} / {len(y_true)}"
    )


print("\n")
print("=" * 50)
print("DIAGNOSTIC COMPLETE")
print("=" * 50)