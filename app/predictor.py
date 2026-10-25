
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_finetuned_best.keras"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

IMAGE_SIZE = (224, 224)

_model = None


def load_model():
    """Load the trained model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )

    return _model


def predict_image(image: Image.Image) -> dict:
    """Predict the class of an uploaded MRI image."""

    if not isinstance(image, Image.Image):
        raise TypeError("Please provide a valid PIL image.")

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    model = load_model()
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            CLASS_NAMES[index]: float(probabilities[index])
            for index in range(len(CLASS_NAMES))
        },
    }
