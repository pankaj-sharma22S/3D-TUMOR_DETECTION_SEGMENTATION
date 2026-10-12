
import re

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
from skimage import measure

from model_loader import load_model
from preprocessing import (
decode_upload,
filter_mri,
sharpen_mri,
preprocess_mri,
predict_mri,
create_overlay,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain MRI Segmentation",
    page_icon="🧠",
    layout="wide",
)

dark_mode = st.sidebar.checkbox("Dark mode", value=True)

if dark_mode:
    page_background = "#0b1220"
    panel_background = "#111c2f"
    text_color = "#e5edf8"
    muted_color = "#9aabc2"
    border_color = "#26364d"
    accent_color = "#4f9cf9"
else:
    page_background = "#f4f7fb"
    panel_background = "#ffffff"
    text_color = "#172033"
    muted_color = "#63718a"
    border_color = "#d8e1ed"
    accent_color = "#2563eb"

st.markdown(
    f"""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {page_background};
        color: {text_color};
    }}
    [data-testid="stSidebar"] {{
        background-color: {panel_background};
        border-right: 1px solid {border_color};
    }}
    [data-testid="stFileUploaderDropzone"] {{
        border: 1px dashed {accent_color};
        border-radius: 12px;
        background-color: {panel_background};
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        border: 1px solid {border_color};
        border-radius: 8px;
    }}
    [data-testid="stMetric"] {{
        background-color: {panel_background};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 1rem;
    }}
    h1, h2, h3, p, label {{
        color: {text_color};
    }}
    [data-testid="stCaptionContainer"] {{
        color: {muted_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Brain MRI Analysis")
st.caption("MRI segmentation and tumor-type prediction")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

with st.container(border=True):
    st.subheader("Upload MRI images")
    upload_mode = st.radio(
        "Choose input",
        options=["Select images", "Select a folder"],
        horizontal=True,
        label_visibility="collapsed",
    )
    uploaded_files = st.file_uploader(
        "Choose MRI image files" if upload_mode == "Select images"
        else "Choose a folder containing MRI images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=(
            True if upload_mode == "Select images" else "directory"
        ),
        help=(
            "Upload one image or multiple slices from the same scan. "
            "A single image is reused for the 3D preview."
        ),
        label_visibility="collapsed",
    )

TUMOR_CLASSES = [
    (1, "Class 1 - NCR/NET", "blue", 0.80),
    (2, "Class 2 - Edema", "green", 0.35),
    (3, "Class 3 - Enhancing Tumor", "red", 1.00),
]

st.sidebar.header("OpenCV Image Filters")

enable_filter = st.sidebar.checkbox(
    "Enable bilateral denoising",
    value=True,
)

sharpen_amount = st.sidebar.slider(
    "Sharpening strength",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
)

overlay_alpha = st.sidebar.slider(
    "Segmentation overlay opacity",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
)

brain_threshold = st.sidebar.slider(
    "Brain surface intensity threshold",
    min_value=0,
    max_value=100,
    value=10,
)

brain_step = st.sidebar.select_slider(
    "3D brain mesh detail",
    options=[1, 2, 3, 4, 5],
    value=3,
)

tumor_step = st.sidebar.select_slider(
    "3D tumor mesh detail",
    options=[1, 2, 3, 4],
    value=2,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def natural_sort_key(filename):
    """Sort slice_2 before slice_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


# --------------------------------------------------
# PROCESS USER-UPLOADED IMAGES
# --------------------------------------------------

if not uploaded_files:
    st.info("Choose one or more MRI images, or select a folder, to begin.")
    st.stop()


# Use filenames to order slices naturally.
# Make sure filenames correspond to the correct slice order.
uploaded_files = sorted(
    uploaded_files,
    key=lambda file: natural_sort_key(file.name),
)

st.write(f"**Uploaded images:** {len(uploaded_files)}")

if len(uploaded_files) > 1:
    st.warning(
        "For meaningful 3D results, all uploads must belong to the "
        "same scan and be in the correct anatomical slice order."
    )


brain_slices = []
mask_slices = []
results = []

progress = st.progress(0, text="Processing MRI images...")

for index, uploaded_file in enumerate(uploaded_files):
    try:
        original = decode_upload(uploaded_file)

        # Model input: keep consistent with training.
        # Do not silently apply new filters to model input.
        resized, model_input = preprocess_mri(
            original,
            size=(128, 128),
        )

        mask, _, _ = predict_mri(
            model,
            model_input,
        )

        # Filters are used for display, not prediction.
        if enable_filter:
            display_image = filter_mri(resized)
        else:
            display_image = resized.copy()

        display_image = sharpen_mri(
            display_image,
            amount=sharpen_amount,
        )

        overlay = create_overlay(
            display_image,
            mask,
            alpha=overlay_alpha,
        )

        results.append({
            "name": uploaded_file.name,
            "original": resized,
            "mask": mask,
            "overlay": overlay,
            "tumor_type": max(
                TUMOR_CLASSES,
                key=lambda tumor_class: np.count_nonzero(
                    mask == tumor_class[0]
                ),
            )[1]
            if np.any(mask > 0)
            else "No tumor region detected",
        })

        # Build volumes using model predictions and display images.
        brain_slices.append(display_image)
        mask_slices.append(mask)

    except Exception as error:
        st.error(f"Error processing {uploaded_file.name}: {error}")

    progress.progress(
        (index + 1) / len(uploaded_files),
        text=f"Processed {index + 1}/{len(uploaded_files)} images",
    )

progress.empty()

if not results:
    st.error("No images could be processed.")
    st.stop()


# --------------------------------------------------
# DISPLAY 2D RESULTS
# --------------------------------------------------

st.header("Prediction")

selected_name = st.selectbox(
    "Choose an MRI slice",
    options=[result["name"] for result in results],
)

selected = next(
    result for result in results
    if result["name"] == selected_name
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.image(
            selected["original"],
            caption="MRI image",
            clamp=True,
            use_container_width=True,
        )

with col2:
    with st.container(border=True):
        st.image(
            selected["overlay"],
            caption="Segmentation preview",
            use_container_width=True,
        )

st.success(f"Predicted tumor type: **{selected['tumor_type']}**")


# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("Download results")

mask_png = cv2.imencode(
    ".png",
    selected["mask"],
)[1].tobytes()

overlay_bgr = cv2.cvtColor(
    selected["overlay"],
    cv2.COLOR_RGB2BGR,
)

overlay_png = cv2.imencode(
    ".png",
    overlay_bgr,
)[1].tobytes()

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        "Download segmentation mask",
        data=mask_png,
        file_name=f"{selected_name.rsplit('.', 1)[0]}_mask.png",
        mime="image/png",
    )

with download_col2:
    st.download_button(
        "Download overlay",
        data=overlay_png,
        file_name=f"{selected_name.rsplit('.', 1)[0]}_overlay.png",
        mime="image/png",
    )


# --------------------------------------------------
# 3D VIEWER FOR MULTIPLE SLICES
# --------------------------------------------------

st.header("3D Preview")

brain_volume = np.stack(brain_slices, axis=0)
tumor_volume = np.stack(mask_slices, axis=0)

if len(results) == 1:
    preview_depth = 96
    source_brain = brain_volume[0]
    source_tumor = tumor_volume[0]
    brain_layers = []
    tumor_layers = []
    height, width = source_brain.shape
    y_grid, x_grid = np.ogrid[:height, :width]
    brain_oval = (
        ((x_grid - (width - 1) / 2) / (width * 0.49)) ** 2
        + ((y_grid - (height - 1) / 2) / (height * 0.49)) ** 2
        <= 1
    )

    for layer in range(preview_depth):
        depth_position = 2 * (layer + 0.5) / preview_depth - 1
        scale = max(0.04, np.sqrt(max(0, 1 - depth_position**2)))
        layer_height = max(1, round(height * scale))
        layer_width = max(1, round(width * scale))

        resized_brain = cv2.resize(
            source_brain,
            (layer_width, layer_height),
            interpolation=cv2.INTER_AREA,
        )
        resized_tumor = cv2.resize(
            source_tumor,
            (layer_width, layer_height),
            interpolation=cv2.INTER_NEAREST,
        )

        brain_layer = np.zeros_like(source_brain)
        tumor_layer = np.zeros_like(source_tumor)
        top = (height - layer_height) // 2
        left = (width - layer_width) // 2
        brain_layer[top:top + layer_height, left:left + layer_width] = resized_brain
        tumor_layer[top:top + layer_height, left:left + layer_width] = resized_tumor
        brain_layer[~brain_oval] = 0
        tumor_layer[~brain_oval] = 0
        brain_layers.append(brain_layer)
        tumor_layers.append(tumor_layer)

    brain_volume = np.stack(brain_layers, axis=0)
    tumor_volume = np.stack(tumor_layers, axis=0)
    st.caption(
        f"Approximate 3D preview built from one image across {preview_depth} "
        "tapered layers. This is a visualization, not additional MRI data."
    )

if np.any(tumor_volume > 0):
    fig = go.Figure()

    brain_binary = np.zeros(brain_volume.shape, dtype=bool)
    for slice_index, brain_slice in enumerate(brain_volume):
        slice_mask = binary_closing(
            brain_slice > brain_threshold,
            structure=np.ones((3, 3), dtype=bool),
            iterations=2,
        )
        slice_mask = binary_fill_holes(slice_mask)
        components, component_count = label(slice_mask)
        if component_count:
            component_sizes = np.bincount(components.ravel())
            component_sizes[0] = 0
            brain_binary[slice_index] = components == component_sizes.argmax()

    if np.any(brain_binary):
        try:
            vertices, faces, _, _ = measure.marching_cubes(
                brain_binary,
                level=0.5,
                step_size=brain_step,
            )
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 2],
                    y=vertices[:, 1],
                    z=vertices[:, 0],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="lightgray",
                    opacity=0.20,
                    name="Brain Anatomy",
                )
            )
        except RuntimeError:
            pass

    for class_id, class_name, color, opacity in TUMOR_CLASSES:
        class_volume = (
            (tumor_volume == class_id) & brain_binary
        ).astype(np.uint8)
        if not np.any(class_volume):
            continue

        try:
            surface_volume = class_volume.astype(np.float32)
            roughness = gaussian_filter(
                np.random.default_rng(class_id).normal(
                    size=class_volume.shape
                ).astype(np.float32),
                sigma=1.0,
            )
            roughness_std = float(roughness.std())
            if roughness_std > 0:
                surface_volume += np.clip(
                    roughness / roughness_std * 0.12,
                    -0.3,
                    0.3,
                )
            vertices, faces, _, _ = measure.marching_cubes(
                surface_volume,
                level=0.5,
                step_size=tumor_step,
            )
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 2],
                    y=vertices[:, 1],
                    z=vertices[:, 0],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=opacity,
                    name=class_name,
                )
            )
        except RuntimeError:
            continue

    fig.update_layout(
        title="3D Multi-Color Tumor Map",
        scene=dict(
            xaxis_title="Width (X)",
            yaxis_title="Height (Y)",
            zaxis_title="Slices (Z)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title="Predicted Classes"),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
else:
    st.info("No tumor region was detected for the 3D preview.")

st.caption(
    "This application visualizes model predictions for research and "
    "educational use. It is not a medical diagnostic system."
)