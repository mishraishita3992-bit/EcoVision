from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load the fine-tuned model
MODEL_PATH = "models/plastic_classifier_finetuned.keras"

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Keep this order exactly the same as during training
CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image using the 'image' field."
        }), 400

    try:
        image_file = request.files["image"]

        # Read image
        image = Image.open(
            io.BytesIO(image_file.read())
        )

        # Convert to RGB
        image = image.convert("RGB")

        # Resize to the size used during training
        image = image.resize((224, 224))

        # Convert image to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = image_array / 127.5 - 1.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Make prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Find highest probability
        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index]
        )

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(
                confidence * 100,
                2
            ),
            "all_scores": {
                CLASS_NAMES[i]: round(
                    float(predictions[i]) * 100,
                    2
                )
                for i in range(len(CLASS_NAMES))
            },
            "note": "Experimental prediction. Verify the resin type before recycling."
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run()