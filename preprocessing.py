
import cv2
import numpy as np


def decode_upload(uploaded_file):
    """Decode an uploaded image into an 8-bit grayscale image."""
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read uploaded image: {uploaded_file.name}")

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.dtype != np.uint8:
        gray = cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    return gray


def resize_mri(image, size=(128, 128)):
    """Resize to the dimensions expected by the trained model."""
    width, height = size

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )


def filter_mri(image, strength=40, spatial_strength=40):
    """Reduce noise while preserving edges."""
    return cv2.bilateralFilter(
        image,
        d=5,
        sigmaColor=strength,
        sigmaSpace=spatial_strength
    )


def sharpen_mri(image, amount=1.0):
    """Sharpen an image using OpenCV unsharp masking."""
    if amount <= 0:
        return image.copy()

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)

    sharpened = cv2.addWeighted(
        image, 1.0 + amount,
        blurred, -amount,
        0
    )

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_mri(image, size=(128, 128)):
    """
    Prepare model input.

    Assumes training used grayscale images resized to 128x128
    and normalized using pixel values / 255.
    """
    resized = resize_mri(image, size)

    model_input = resized.astype(np.float32) / 255.0
    model_input = model_input[np.newaxis, ..., np.newaxis]

    return resized, model_input


def predict_mri(model, model_input):
    """Predict the class of every pixel."""
    probabilities = model.predict(model_input, verbose=0)

    if probabilities.ndim != 4 or probabilities.shape[0] != 1:
        raise ValueError(
            f"Unexpected model output shape: {probabilities.shape}"
        )

    if probabilities.shape[-1] != 4:
        raise ValueError(
            f"Expected 4 output classes, got {probabilities.shape[-1]}."
        )

    mask = np.argmax(probabilities[0], axis=-1).astype(np.uint8)
    pixel_confidence = np.max(probabilities[0], axis=-1)
    mean_confidence = float(pixel_confidence.mean())

    return mask, probabilities[0], mean_confidence


def colorize_mask(mask):
    """Convert class labels into a BGR color image."""
    colors = {
        0: (0, 0, 0),          # Background
        1: (255, 120, 40),     # Class 1: blue-ish
        2: (80, 220, 80),      # Class 2: green
        3: (40, 70, 255),      # Class 3: red
    }

    output = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8
    )

    for class_id, color in colors.items():
        output[mask == class_id] = color

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def create_overlay(image, mask, alpha=0.45):
    """Overlay predicted foreground classes on the MRI."""
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = colorize_mask(mask)
    colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)

    blended = cv2.addWeighted(base, 1 - alpha, colors, alpha, 0)

    blended[mask == 0] = base[mask == 0]

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)