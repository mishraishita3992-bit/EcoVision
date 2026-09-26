import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/plastic_classifier.keras"

IMAGE_PATH = r"C:\Users\KIIT\OneDrive\Desktop\EcoVision\dataset\train\PET\PET_1.jpg"
CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP"]

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Load image
image = Image.open(IMAGE_PATH)

print("Original image size:", image.size)

# Convert to RGB
image = image.convert("RGB")

# Resize
image = image.resize((224, 224))

# Convert to NumPy
image_array = np.array(image, dtype=np.float32)

# Normalize exactly like training
image_array = image_array / 127.5 - 1.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Prediction
predictions = model.predict(image_array, verbose=0)[0]

print("\n==============================")
print("DIRECT MODEL PREDICTION")
print("==============================")

for i, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {predictions[i] * 100:.2f}%")

predicted_index = np.argmax(predictions)

print("\nPredicted class:", CLASS_NAMES[predicted_index])
print("Confidence:", f"{predictions[predicted_index] * 100:.2f}%")