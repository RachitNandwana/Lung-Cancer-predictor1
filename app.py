import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import tempfile

st.set_page_config(page_title="Lung Cancer Predictor", layout="wide")
st.title("Lung Cancer Predictor")

# Local dataset image path provided in conversation (use as placeholder file URL / integration)
DATASET_IMAGE_LOCAL_PATH = "/mnt/data/7fff4457-81ee-4946-abe9-827fd7ca287b.png"

# Local model filename expected in same repo as app.py
LOCAL_MODEL_PATH = Path("lung_cancer_model.pkl")

# ----------------- Model feature order (MUST match training order) -----------------
FEATURE_ORDER = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
    "Anxiety + yellow fingers",
]

# ----------------- Helpers -----------------
def load_local_model(path: Path):
    """Load a local joblib model. Returns (model, error_message_or_None)."""
    if not path.exists():
        return None, f"Model file not found at: {path.resolve()}"
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

def interpret_and_show_prediction(pred):
    """
    Interpret model output and present a user-friendly message.
    - If output in [0,1] treat it as probability.
    - Otherwise use 0.5 threshold as fallback.
    """
    try:
        pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
    except Exception:
        st.error("Model returned an unrecognized value.")
        return

    # If model returns a probability-like number in [0,1]
    if 0.0 <= pred_value <= 1.0:
        pct = pred_value * 100.0
        if pred_value >= 0.5:
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background:#3b0b0b;color:#fff'>"
                f"<strong>High chance of lung cancer</strong>.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background:#083a1f;color:#fff'>"
                f"<strong>Low chance of lung cancer</strong>.</div>",
                unsafe_allow_html=True,
            )
    else:
        # Non-probabilistic output: fallback
        if pred_value >= 0.5:
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background:#3b0b0b;color:#fff'>"
                f"<strong>High chance of lung cancer</strong> — model output: <strong>{pred_value:.3f}</strong>.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background:#083a1f;color:#fff'>"
                f"<strong>Low chance of lung cancer</strong> — model output: <strong>{pred_value:.3f}</strong>.</div>",
                unsafe_allow_html=True,
            )

    # Professional disclaimer (always shown)
    st.warning("**Disclaimer:** Automated prediction may vary. This is NOT a medical diagnosis. Please consult a qualified healthcare professional for medical advice.")

# ----------------- App UI -----------------
st.sidebar.header("Mode")
mode = st.sidebar.radio("", ["Patient", "Doctor"])

# ----------- PATIENT MODE -----------
# ---------- Patient Mode (clean headings, only symptom chips selectable) ----------
if mode == "Patient":
    st.header("Patient Mode")
    st.write("Select symptoms you have. Selected = 1 (Yes), Unselected = 0 (No).")

    FEATURE_LABEL_MAP = {
        "Smoking": "SMOKING",
        "Yellow fingers": "YELLOW_FINGERS",
        "Anxiety": "ANXIETY",
        "Peer pressure": "PEER_PRESSURE",
        "Chronic disease": "CHRONIC_DISEASE",
        "Fatigue": "FATIGUE",
        "Allergy": "ALLERGY",
        "Wheezing": "WHEEZING",
        "Alcohol consuming": "ALCOHOL_CONSUMING",
        "Coughing": "COUGHING",
        "Shortness of breath": "SHORTNESS_OF_BREATH",
        "Swallowing difficulty": "SWALLOWING_DIFFICULTY",
        "Chest pain": "CHEST_PAIN",
    }

    GROUPS = {
        "Behavior & Exposures": [
            "Smoking", "Alcohol consuming", "Peer pressure", "Yellow fingers"
        ],
        "Comorbidities / General": [
            "Chronic disease", "Fatigue", "Allergy"
        ],
        "Respiratory / ENT": [
            "Wheezing", "Coughing", "Shortness of breath", "Swallowing difficulty", "Chest pain"
        ],
        "Psych / Other": [
            "Anxiety"
        ],
    }

    with st.form("patient_form"):
        selected_labels = []

        # Two-column layout WITHOUT dropdowns/expanders
        cols = st.columns(2)
        for i, (group_name, labels) in enumerate(GROUPS.items()):
            col = cols[i % 2]

            # Just a title — not clickable
            col.markdown(f"### {group_name}")

            # Only the symptom chips are selectable
            chosen = col.multiselect(
                label="",  # no label so only chips show
                options=labels,
                default=[]
            )

            for ch in chosen:
                if ch not in selected_labels:
                    selected_labels.append(ch)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Convert selections to 0/1 inputs
        inputs = {}
        for label, key in FEATURE_LABEL_MAP.items():
            inputs[key] = 1 if label in selected_labels else 0

        # Derived column
        inputs["Anxiety + yellow fingers"] = 1 if (
            inputs["ANXIETY"] == 1 or inputs["YELLOW_FINGERS"] == 1
        ) else 0

        # Build feature vector
        x = np.array([inputs[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)

        model, err = load_local_model(LOCAL_MODEL_PATH)
        if model is None:
            st.error(f"Model load failed: {err}")
        else:
            try:
                pred = model.predict(x)
                interpret_and_show_prediction(pred)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ----------- DOCTOR MODE -----------
# ---------- DOCTOR MODE (CSV-based login) ----------
# ---------- DOCTOR MODE: download .h5 from Dropbox, load it, run prediction on uploaded CT image ----------
# Replace your existing Doctor Mode block with this code.
# Behavior:
# - Expects a Dropbox share link (the one you gave will be converted to dl=1)
# - Downloads the .h5 model (streamed & cached), loads it (cached), then preprocesses the uploaded image
# - Uses model.input_shape to pick resize dims when possible; falls back to (224,224)
# - Runs model.predict and shows a simple Low/High chance message (no probability), plus disclaimer
# - If no file uploaded, uses developer-provided example local image path for testing:
#     /mnt/data/7fff4457-81ee-4946-abe9-827fd7ca287b.png

import streamlit as st
from pathlib import Path
import tempfile
import os
import requests
from PIL import Image
import numpy as np
import tensorflow as tf

# Developer-provided dataset image local path (from conversation history)
DEV_DATASET_IMAGE_PATH = "/mnt/data/7fff4457-81ee-4946-abe9-827fd7ca287b.png"

# Default Dropbox URL (user-provided)
DEFAULT_DROPBOX_URL = "https://www.dropbox.com/scl/fi/bab21h5uesf59j2yn9ngu/lung_cancer_model_finalmodel.h5?rlkey=uw8n849or6sfcolvevbjngahh&st=cp4zn6qy&dl=0"

# Helpers: convert Dropbox share to direct-download and stream-download large file (cached)
def dropbox_to_direct(url: str) -> str:
    if not url:
        return url
    if "dropbox.com" in url:
        if "dl=1" in url:
            return url
        if "dl=0" in url:
            return url.replace("dl=0", "dl=1")
        if "?" in url:
            return url + "&dl=1"
        return url + "?dl=1"
    return url

@st.cache_data(show_spinner=False)
def download_large_file(url: str, target_path: str, chunk_size: int = 1024 * 1024):
    # If url is a local path, return as-is
    if os.path.exists(url):
        return url
    # If file already downloaded, return early
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path
    # Stream download
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return target_path

@st.cache_resource
def load_keras_model_from_path(local_path: str):
    # load model (compile=False to speed up and avoid optimizer issues)
    model = tf.keras.models.load_model(local_path, compile=False)
    return model

def preprocess_image_for_model(image_path: str, model_input_shape):
    """
    Loads image from image_path and preprocesses to fit model_input_shape.
    model_input_shape: tuple, e.g. (None, height, width, channels) or (None, channels, height, width)
    Returns numpy array of shape (1, H, W, C) and dtype float32 scaled to [0,1].
    """
    # Determine (H, W, C)
    # model_input_shape may be e.g. (None, 224, 224, 3) or (None, 3, 224, 224)
    if model_input_shape is None or len(model_input_shape) < 3:
        target_size = (224, 224)
        channels = 3
    else:
        # remove batch dim if present
        dims = list(model_input_shape)
        if dims[0] is None:
            dims = dims[1:]
        # detect channel-last
        if len(dims) == 3:
            if dims[-1] in (1, 3):
                height, width, channels = dims
            elif dims[0] in (1, 3):
                channels, height, width = dims
            else:
                # fallback
                height, width = dims[0], dims[1]
                channels = 3
        else:
            # fallback
            height, width = 224, 224
            channels = 3
        target_size = (int(width), int(height))  # PIL expects (width, height)

    # Open image, convert to RGB if model expects 3 channels else L
    img = Image.open(image_path)
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    # Resize to target_size (width,height)
    img = img.resize(target_size, Image.ANTIALIAS)
    arr = np.array(img).astype("float32") / 255.0
    # If model expects channel-first, transpose
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] != (channels if channels is not None else arr.shape[-1]):
        # try to adapt channels (e.g., grayscale -> 3 channels)
        if arr.shape[-1] == 1 and channels == 3:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 3 and channels == 1:
            arr = np.mean(arr, axis=-1, keepdims=True)
    # Ensure shape is (H,W,C)
    if arr.shape[-1] != channels and channels is not None:
        # final safety: reshape or pad/truncate channels
        if arr.shape[-1] > channels:
            arr = arr[..., :channels]
        else:
            while arr.shape[-1] < channels:
                arr = np.concatenate([arr, arr[..., :1]], axis=-1)
    # Expand batch dim
    batch = np.expand_dims(arr, axis=0)
    return batch

# ---------------- UI & Logic ----------------
st.header("Doctor Mode — CT analysis (Dropbox-hosted model)")

st.info("Paste your Dropbox share link (or leave default). Upload a CT image and click 'Run Image Analysis' to get prediction. Image analysis uses the provided .h5 model downloaded from Dropbox.")

# Input: model URL (pre-filled with the Dropbox link you provided)
model_url_input = st.text_input("Model URL (Dropbox share link)", value=DEFAULT_DROPBOX_URL)

# Upload CT scan (doctor)
uploaded_file = st.file_uploader("Upload CT scan image (DICOM/JPEG/PNG). If DICOM, upload as .dcm", type=["png", "jpg", "jpeg", "dcm"])

# Save uploaded file to temp path if present, else we will use developer-provided dataset image path
uploaded_local_path = None
if uploaded_file is not None:
    try:
        suffix = Path(uploaded_file.name).suffix or ".img"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()
        uploaded_local_path = tmp_path
        st.success("Uploaded file saved temporarily.")
        # try to preview if image
        try:
            im = Image.open(uploaded_local_path)
            st.image(im, caption="Uploaded CT preview", use_column_width=True)
        except Exception:
            st.info("Preview not available (file might be DICOM).")
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")

# Button to run analysis
if st.button("Run Image Analysis"):
    # Determine which image path to use
    image_path_to_use = uploaded_local_path if uploaded_local_path else DEV_DATASET_IMAGE_PATH
    if not os.path.exists(image_path_to_use):
        st.error("No image available to analyze. Upload a CT scan or ensure developer dataset path exists.")
    else:
        # Prepare model URL (convert Dropbox share to direct)
        direct_model_url = dropbox_to_direct(model_url_input.strip())
        # Prepare local path to save model
        safe_name = Path(direct_model_url).name or "downloaded_model.h5"
        tmp_dir = Path(tempfile.gettempdir()) / "models"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = str(tmp_dir / safe_name)

        # Download model (or use local path if provided)
        try:
            with st.spinner("Downloading model (if not already cached)..."):
                model_file = download_large_file(direct_model_url, local_model_path)
            st.success("Model file available.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            model_file = None

        if model_file:
            # Load model (cached)
            try:
                with st.spinner("Loading model into memory..."):
                    model = load_keras_model_from_path(model_file)
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                model = None

            if model is not None:
                # Get model input shape if available
                try:
                    model_input_shape = None
                    # tf.keras models may have .input_shape or .inputs[0].shape
                    if hasattr(model, "input_shape"):
                        model_input_shape = model.input_shape
                    elif hasattr(model, "inputs") and len(model.inputs) > 0:
                        model_input_shape = tuple(model.inputs[0].shape.as_list())
                except Exception:
                    model_input_shape = None

                # Preprocess image
                try:
                    x = preprocess_image_for_model(image_path_to_use, model_input_shape)
                except Exception as e:
                    st.error(f"Failed to preprocess image: {e}")
                    x = None

                if x is not None:
                    # Predict
                    try:
                        pred = model.predict(x)
                        # Interpret prediction using 0.5 threshold (no probabilities displayed)
                        try:
                            pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
                        except Exception:
                            st.error("Model returned an unrecognized prediction.")
                            pred_value = None

                        if pred_value is not None:
                            if pred_value >= 0.5:
                                st.markdown(
                                    "<div style='padding:12px;border-radius:8px;background:#3b0b0b;color:#fff'>"
                                    "<strong>High chance of lung cancer</strong></div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    "<div style='padding:12px;border-radius:8px;background:#083a1f;color:#fff'>"
                                    "<strong>Low chance of lung cancer</strong></div>",
                                    unsafe_allow_html=True,
                                )
                            st.warning("**Disclaimer:** Automated prediction may vary. This is NOT a medical diagnosis. Please consult a qualified healthcare professional for medical advice.")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

