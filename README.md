<h1 align="center">Brain Tumor Segmentation with U-Net</h1>

<p align="center">
  <strong>Pixel-wise MRI segmentation with an interactive Streamlit viewer</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow and Keras">
  <img src="https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Research%20prototype-Not%20for%20diagnosis-6B7280?style=for-the-badge" alt="Research prototype, not for diagnosis">
</p>

<p align="center">
  <img src="src/Screenshot%202026-10-12%20211436.png" alt="Interactive 3D brain tumor segmentation preview" width="800">
</p>

## Project overview

This project explores **semantic segmentation of brain MRI slices** using a U-Net model built with TensorFlow/Keras. Given a preprocessed grayscale MRI image, the model predicts a class for each pixel. The Streamlit application presents the input alongside a colored segmentation overlay and provides an interactive 3D visualization.

The workflow is intended for learning and experimentation: prepare paired MRI images and masks, train and evaluate a segmentation model in the notebook, then inspect predictions in the app. It is **not clinically validated** and must not be used to diagnose or guide treatment.

## What it does

- Trains a U-Net encoder-decoder with skip connections for four-class pixel labeling.
- Prepares grayscale MRI slices and segmentation masks at `128 × 128`.
- Reports training and evaluation metrics in the notebook.
- Accepts one or more image files, or a folder of images, in the Streamlit UI.
- Shows the selected MRI and its predicted segmentation overlay.
- Builds colored 3D meshes from predicted labels for visualization.
- Supports dark and light UI themes and optional display filters.

### Segmentation labels

The notebook remaps source mask label `4` to model index `3`. Confirm this mapping against the dataset and preprocessing used for your own run.

| Model index | Source mask label | Region |
|---:|---:|---|
| **0** | 0 | Background |
| **1** | 1 | NCR/NET |
| **2** | 2 | Edema |
| **3** | 4 | Enhancing tumor |

## Project structure

The repository currently uses root-level Python and notebook files; `src/` contains viewer screenshots.

```text
CNN_BRAIN_TUMOR/
├── app.py                         # Streamlit interface and 3D viewer
├── model_loader.py                # Loads the trained Keras model
├── preprocessing.py               # Image decoding, preprocessing, prediction helpers
├── cnn_tumor.ipynb                # Training, evaluation, and notebook visualizations
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignores virtual environments and generated files
├── .gitattributes                 # Keeps notebook line endings as LF
└── src/
    ├── Screenshot 2026-10-12 210557.png
    ├── Screenshot 2026-10-12 211436.png
    └── image.png
```

The trained model file is **not stored in this repository**. `model_loader.py` currently expects it at `E:\best_unet_model.keras`.

## Run the Streamlit app

Use Python 3.11 and run these commands from the repository root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Before starting the app, make the trained model available at `E:\best_unet_model.keras`, or update `MODEL_PATH` in `model_loader.py` to point to your local `.keras` file. Keep large model files and private MRI data out of source control.

Upload one image to view a prediction, or upload a set of ordered slices from the **same scan** for a more meaningful multi-slice visualization. The app orders uploads by filename; name slices consistently (for example, `slice_001.png`, `slice_002.png`, ...).

## Run the training notebook

Open [`cnn_tumor.ipynb`](cnn_tumor.ipynb) in Jupyter or Google Colab. Follow its cells in order for dataset setup, preprocessing, splitting, training, evaluation, and visualization. The notebook contains environment-specific dataset and Drive paths; update those paths for your own setup. Store access credentials in a secure environment or notebook secret store, never directly in the notebook.

The notebook configures a checkpoint named `best_unet_model.keras`. Copy or download the resulting model to the path expected by `model_loader.py` before launching the app.

## Model and training details

| Setting | Value |
|---|---|
| **Architecture** | U-Net encoder-decoder with skip connections |
| **Framework** | TensorFlow / Keras |
| **Input** | `128 × 128 × 1` grayscale slice |
| **Output** | Four-class pixel-wise segmentation |
| **Optimizer** | Adam (`0.0001` learning rate) |
| **Loss** | Categorical cross-entropy |
| **Batch size** | `32` |
| **Epoch limit** | `10` |
| **Checkpoint** | `best_unet_model.keras`, selected by validation loss |
| **Early stopping** | Validation-loss patience of `2` |
| **Parameters** | `7,707,652` total (`7,702,020` trainable) |

### Preprocessing

As configured in the notebook, images and masks are resized to `128 × 128`; image slices receive a `5 × 5` Gaussian blur and are normalized to `[0, 1]`; source mask label `4` is mapped to model index `3`; and masks are converted to four-class one-hot tensors. The app's prediction preprocessing should remain consistent with the preprocessing used to train the checkpoint.

## Reported notebook results

The values below are copied from the notebook outputs. They describe the reported aggregate pixel-level evaluation, **not** class-wise tumor overlap or clinical effectiveness.

| Evaluation | Loss | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| **Test set** | `0.0346` | **99.30%** | **99.36%** | **99.25%** |
| **Validation, epoch 10** | `0.0346` | **99.31%** | **99.37%** | **99.26%** |

The notebook reports **11,163 paired image-mask slices**: 7,814 training, 1,674 validation, and 1,675 test. Its split is performed over shuffled slices, not demonstrated patient-level groups. Slices from one patient may therefore appear across different splits, which can make evaluation optimistic. Patient-level separation and per-class Dice/IoU should be used for stronger validation.

## Understanding the 3D preview

For multiple uploads, the app stacks the predicted masks and MRI slices into a volume and extracts surfaces for the brain and tumor classes. The best input is a correctly ordered set of slices from one scan, with consistent acquisition and orientation.

When only one image is provided, the app creates a **synthetic tapered preview** by repeating and resizing that slice across depth. It can help demonstrate the viewer, but it does not recover unseen anatomy or represent a true 3D scan. The output is sensitive to slice ordering, intensity thresholding, and model segmentation quality.

## Limitations

- This is an experimental educational project, **not a medical device**.
- Aggregate pixel accuracy can be dominated by background pixels and does not establish tumor-region quality.
- The notebook's slice-level split may not measure generalization to unseen patients.
- Verify class labels, data preprocessing, and dataset version before interpreting predictions.
- Realistic 3D interpretation requires correctly ordered source slices and spatial metadata.
- Evaluation should include patient-level separation, per-class Dice/IoU, sensitivity, specificity, and visual failure-case review.

## Data handling and attribution

- Do not commit raw MRI scans, patient-identifying information, credentials, or API tokens.
- Check the applicable BraTS 2021 access terms and citation requirements before redistributing data.
- If a secret has been committed, revoke or rotate it and remove it from repository history as appropriate.

The notebook references the **BraTS 2021** dataset. Cite the dataset and any additional models, papers, or resources used in your own work according to their terms.

---

<p align="center"><strong>For research and education only — not for diagnosis or treatment decisions.</strong></p>
