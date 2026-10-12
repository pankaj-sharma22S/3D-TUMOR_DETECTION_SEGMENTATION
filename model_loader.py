from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path(r"E:\best_unet_model.keras")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )