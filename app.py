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
# --- Put this inside your Doctor Mode block (replace the old Run Image Analysis handler) ---

# Hidden/internal Dropbox link (not shown to users)
MODEL_URL = "https://www.dropbox.com/scl/fi/bab21h5uesf59j2yn9ngu/lung_cancer_model_finalmodel.h5?rlkey=uw8n849or6sfcolvevbjngahh&st=cp4zn6qy&dl=0"

# Developer-provided fallback image local path (use this if doctor didn't upload an image)
DEV_DATASET_IMAGE_PATH = "/mnt/data/bc397ec8-5d3e-4aa5-a0a2-533368b0dc44.png"

# Convert to direct-download (reuse your dropbox_to_direct function)
direct_model_url = dropbox_to_direct(MODEL_URL)

# When doctor clicks the button, use uploaded file if present; otherwise fallback to DEV path.
if st.button("Run Image Analysis"):
    # choose image path
    image_path_to_use = uploaded_local_path if uploaded_local_path else DEV_DATASET_IMAGE_PATH

    if not os.path.exists(image_path_to_use):
        st.error("No image available to analyze. Upload a CT scan or ensure the developer dataset path exists.")
    else:
        # prepare local model path
        safe_name = Path(direct_model_url).name or "downloaded_model.h5"
        tmp_dir = Path(tempfile.gettempdir()) / "models"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = str(tmp_dir / safe_name)

        # download model (cached)
        try:
            with st.spinner("Downloading model (if not cached)..."):
                model_file = download_large_file(direct_model_url, local_model_path)
            st.success("Model file is ready.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            model_file = None

        if model_file:
            try:
                with st.spinner("Loading model into memory..."):
                    model = load_keras_model_from_path(model_file)
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                model = None

            if model is not None:
                # get input shape
                try:
                    model_input_shape = getattr(model, "input_shape", None)
                    if model_input_shape is None and hasattr(model, "inputs") and len(model.inputs) > 0:
                        model_input_shape = tuple(model.inputs[0].shape.as_list())
                except Exception:
                    model_input_shape = None

                # preprocess
                try:
                    x = preprocess_image_for_model(image_path_to_use, model_input_shape)
                except Exception as e:
                    st.error(f"Failed to preprocess image: {e}")
                    x = None

                if x is not None:
                    try:
                        pred = model.predict(x)
                        # interpret using threshold 0.5 only (no probability shown)
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
                            st.warning("**Disclaimer:** Automated prediction may vary. This is NOT a medical diagnosis. Please consult a qualified healthcare professional.")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

