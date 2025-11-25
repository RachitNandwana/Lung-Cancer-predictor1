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
# ------------------- DOCTOR MODE (complete block) -------------------
# Replace your current Doctor Mode block with this entire code section.
# - CSV-based doctor login (doctors.csv at repo root or upload via UI)
# - Upload CT image (or use developer-provided fallback file path)
# - Downloads .h5 from Dropbox (hidden MODEL_URL), streams & caches it
# - Lazy-loads TensorFlow only when needed and shows a clear error if TF is not installed
# - Preprocesses the uploaded image to match model input shape (best-effort) and runs prediction
# - Shows simple High / Low chance message (no probabilities) + disclaimer

import streamlit as st
import pandas as pd
import tempfile
import requests
import os
from pathlib import Path
from PIL import Image
import numpy as np

# ---------------- CONFIG ----------------
# Hidden/internal Dropbox link (not shown to users)
MODEL_URL = "https://www.dropbox.com/scl/fi/bab21h5uesf59j2yn9ngu/lung_cancer_model_finalmodel.h5?rlkey=uw8n849or6sfcolvevbjngahh&st=cp4zn6qy&dl=0"

# Use the most recent uploaded dataset image path (developer-provided) as fallback
DEV_DATASET_IMAGE_PATH = "/mnt/data/bc397ec8-5d3e-4aa5-827fd7ca287b.png"
# Note: per session history a different file path was used in some messages.
# If you're using the file that was uploaded most recently, that path is:
FALLBACK_UPLOADED_IMAGE_PATH = "/mnt/data/bc397ec8-5d3e-4aa5-a0a2-533368b0dc44.png"

# Prefer the latest uploaded file path as fallback
if Path(FALLBACK_UPLOADED_IMAGE_PATH).exists():
    DEV_DATASET_IMAGE_PATH = FALLBACK_UPLOADED_IMAGE_PATH

DOCTORS_CSV_PATH = Path("doctors.csv")  # expected CSV at repo root

# ---------------- Utilities ----------------
def load_doctors_from_csv(path: Path):
    """Return dict {doctor_id: password} or (None, err)"""
    if not path.exists():
        return None, f"doctors CSV not found at {path.resolve()}"
    try:
        df = pd.read_csv(path)
        if "doctor_id" not in df.columns or "password" not in df.columns:
            return None, "CSV must contain 'doctor_id' and 'password' columns"
        return dict(zip(df["doctor_id"].astype(str), df["password"].astype(str))), None
    except Exception as e:
        return None, str(e)

def dropbox_to_direct(url: str) -> str:
    """Convert Dropbox share link to direct-download (dl=1)."""
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
    """
    Stream-download a large file to target_path. If url is a local path, return as-is.
    Cached by URL so repeated calls won't re-download.
    """
    # If url is a local path, return it
    if os.path.exists(url):
        return url

    # If file already downloaded, return
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
    """
    Lazy import TensorFlow and load the Keras model.
    If TensorFlow isn't installed this will raise ModuleNotFoundError which we catch in calling code.
    """
    # Lazy import to avoid ModuleNotFoundError at module import time
    import tensorflow as tf  # may raise ModuleNotFoundError if not installed
    model = tf.keras.models.load_model(local_path, compile=False)
    return model

def preprocess_image_for_model(image_path: str, model_input_shape):
    """
    Load image and preprocess to match model_input_shape as best-effort.
    Returns numpy array shaped (1, H, W, C) with float32 values in [0,1].
    """
    # Determine target size and channels
    target_size = (224, 224)
    channels = 3

    try:
        if model_input_shape:
            dims = list(model_input_shape)
            # remove batch dim if present
            if dims and (dims[0] is None or isinstance(dims[0], int)):
                if dims[0] is None:
                    dims = dims[1:]
            if len(dims) == 3:
                # channel-last (H,W,C)
                if dims[-1] in (1, 3):
                    height, width, channels = int(dims[0]), int(dims[1]), int(dims[2])
                # channel-first (C,H,W)
                elif dims[0] in (1, 3):
                    channels, height, width = int(dims[0]), int(dims[1]), int(dims[2])
                else:
                    height, width = int(dims[0]), int(dims[1])
                target_size = (width, height)  # PIL uses (width, height)
    except Exception:
        # fallback to default target_size / channels
        target_size = (224, 224)
        channels = 3

    # Load image
    img = Image.open(image_path)
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize(target_size, Image.ANTIALIAS)
    arr = np.array(img).astype("float32") / 255.0

    # Ensure channel dimension present
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    # If grayscale but model expects 3 channels, replicate channels
    if arr.shape[-1] == 1 and channels == 3:
        arr = np.repeat(arr, 3, axis=-1)
    # If model expects 1 channel but arr has 3, convert to grayscale
    if arr.shape[-1] == 3 and channels == 1:
        arr = np.mean(arr, axis=-1, keepdims=True)

    # Final safety: adjust channels
    if arr.shape[-1] != channels:
        if arr.shape[-1] > channels:
            arr = arr[..., :channels]
        else:
            while arr.shape[-1] < channels:
                arr = np.concatenate([arr, arr[..., :1]], axis=-1)

    batch = np.expand_dims(arr, axis=0)
    return batch

# ---------------- Session state: login ----------------
if "doctor_logged_in" not in st.session_state:
    st.session_state["doctor_logged_in"] = False
if "doctor_id" not in st.session_state:
    st.session_state["doctor_id"] = None
if "doctors_dict" not in st.session_state:
    st.session_state["doctors_dict"] = None

st.header("Doctor Mode — login required")

# Load doctors.csv if present
if st.session_state["doctors_dict"] is None:
    if DOCTORS_CSV_PATH.exists():
        docs, err = load_doctors_from_csv(DOCTORS_CSV_PATH)
        if docs is None:
            st.error(f"Failed to load doctors.csv: {err}")
            st.info("You may upload a doctors.csv below.")
        else:
            st.session_state["doctors_dict"] = docs
    else:
        st.info("No doctors.csv found. You can upload one below to enable login.")

# Upload doctors.csv (to create or replace)
with st.expander("Upload / Replace doctors.csv (CSV columns: doctor_id,password)", expanded=False):
    uploaded_docs = st.file_uploader("Upload doctors.csv", type=["csv"], key="doctors_csv_upload")
    if uploaded_docs is not None:
        try:
            df_new = pd.read_csv(uploaded_docs)
            if "doctor_id" not in df_new.columns or "password" not in df_new.columns:
                st.error("CSV must contain 'doctor_id' and 'password' columns.")
            else:
                df_new.to_csv(DOCTORS_CSV_PATH, index=False)
                st.success(f"Saved doctors.csv to {DOCTORS_CSV_PATH.resolve()}. Reloading credentials...")
                docs, err = load_doctors_from_csv(DOCTORS_CSV_PATH)
                if docs is not None:
                    st.session_state["doctors_dict"] = docs
                else:
                    st.error(f"Failed to load uploaded CSV: {err}")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# ---------- Login form ----------
if not st.session_state["doctor_logged_in"]:
    with st.form("doctor_login_form"):
        doc_id = st.text_input("Doctor ID")
        doc_pw = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        if st.session_state["doctors_dict"] is None:
            st.error("No credentials available. Upload a doctors.csv or add one in the repo root.")
        else:
            creds = st.session_state["doctors_dict"]
            if doc_id in creds and creds[doc_id] == doc_pw:
                st.session_state["doctor_logged_in"] = True
                st.session_state["doctor_id"] = doc_id
                st.success(f"Logged in as {doc_id}")
                st.experimental_rerun()
            else:
                st.error("Invalid Doctor ID or password.")

# ---------- Doctor dashboard (after login) ----------
else:
    st.markdown(f"**Logged in as:** `{st.session_state['doctor_id']}`")
    if st.button("Logout"):
        st.session_state["doctor_logged_in"] = False
        st.session_state["doctor_id"] = None
        st.experimental_rerun()

    st.header("CT scan upload & analysis")
    st.write("Upload a CT image (DICOM/JPEG/PNG). Click **Run Image Analysis** to analyze the image using the model.")

    uploaded_file = st.file_uploader("Upload CT scan image (DICOM/JPEG/PNG)", type=["png", "jpg", "jpeg", "dcm"], key="doctor_ct_upload")

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
            # Preview if possible
            try:
                im = Image.open(uploaded_local_path)
                st.image(im, caption="Uploaded CT preview", use_column_width=True)
            except Exception:
                st.info("Preview not available (file may be DICOM).")
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
    else:
        st.info("No CT uploaded yet. You may upload or use the developer dataset preview.")

    # Clean UI: do not display the Dropbox link or large blue info box to users
    st.subheader("Dataset preview (developer-provided)")
    try:
        img = Image.open(DEV_DATASET_IMAGE_PATH)
        st.image(img, caption="Dataset preview image", use_column_width=True)
        st.write("Developer-provided local dataset image path (for integration):")
        st.code(DEV_DATASET_IMAGE_PATH)
    except Exception:
        st.warning("Developer dataset preview not available on the filesystem.")

    # ---------------- Run analysis ----------------
    direct_model_url = dropbox_to_direct(MODEL_URL)  # use internal MODEL_URL (hidden from UI)

    if st.button("Run Image Analysis"):
        # choose image to use
        image_path_to_use = uploaded_local_path if uploaded_local_path else DEV_DATASET_IMAGE_PATH

        if not os.path.exists(image_path_to_use):
            st.error("No image available to analyze. Upload a CT scan or ensure the developer dataset path exists.")
        else:
            # prepare deterministic local model filename from URL
            safe_name = Path(direct_model_url).name or "downloaded_model.h5"
            tmp_dir = Path(tempfile.gettempdir()) / "models"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            local_model_path = str(tmp_dir / safe_name)

            # Download model (cached)
            try:
                with st.spinner("Downloading model (if not already cached)..."):
                    model_file = download_large_file(direct_model_url, local_model_path)
                st.success("Model file ready.")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                model_file = None

            if model_file:
                # Attempt to load TensorFlow lazily and load model
                try:
                    with st.spinner("Loading model into memory..."):
                        model = load_keras_model_from_path(model_file)
                    st.success("Model loaded into memory.")
                except ModuleNotFoundError:
                    st.error(
                        "TensorFlow is not installed in this environment. To enable image prediction, add 'tensorflow' to requirements.txt, commit, and redeploy. "
                        "Alternatively host the model on a separate inference server and call it via HTTP."
                    )
                    st.stop()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    model = None

                if model is not None:
                    # Try to get model input shape
                    try:
                        model_input_shape = getattr(model, "input_shape", None)
                        if model_input_shape is None and hasattr(model, "inputs") and len(model.inputs) > 0:
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
                            # Interpret using 0.5 threshold (no probabilities shown)
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


