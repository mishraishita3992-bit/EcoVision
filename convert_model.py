
import tensorflow as tf
from pathlib import Path

MODEL_PATH = "models/plastic_classifier_v2.keras"
OUTPUT_PATH = "models/plastic_classifier_v2.tflite"

print("Loading fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

Path("models").mkdir(exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("Conversion complete!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Model size: {len(tflite_model) / (1024 * 1024):.2f} MB")