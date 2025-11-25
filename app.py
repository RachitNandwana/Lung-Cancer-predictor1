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
else:
    st.header("Doctor Mode — CT scan upload (image analysis not integrated)")
    st.write("Doctor mode currently accepts CT scans (or images) for later processing. Click 'Run Image Analysis' to get the local file path; analysis is not integrated yet.")

    uploaded_file = st.file_uploader("Upload CT scan image (DICOM/JPEG/PNG)", type=["png", "jpg", "jpeg", "dcm"])

    # Two columns: left shows uploaded preview / temp path; right shows dataset preview and its local path
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        if uploaded_file is not None:
            st.success("File uploaded.")
            try:
                suffix = Path(uploaded_file.name).suffix or ".img"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_file.getbuffer())
                tmp.flush()
                tmp_path = tmp.name
                tmp.close()
                st.write("Temporary local file path (use this for integration):")
                st.code(tmp_path)
                # Preview if readable by PIL
                try:
                    im = Image.open(tmp_path)
                    st.image(im, caption="Uploaded image preview", use_column_width=True)
                except Exception:
                    st.info("Preview not available (file may be DICOM or unsupported).")
            except Exception as e:
                st.error(f"Failed to save uploaded file: {e}")
        else:
            st.info("No file uploaded. Use the dataset preview on the right as example.")

    with col2:
        st.subheader("Dataset preview (developer-provided)")
        try:
            img = Image.open(DATASET_IMAGE_LOCAL_PATH)
            st.image(img, caption="Dataset preview image", use_column_width=True)
            st.write("Developer local dataset image path (use as file URL for integration):")
            st.code(DATASET_IMAGE_LOCAL_PATH)
        except Exception:
            st.warning("Dataset preview image not available at expected path.")

    st.markdown("---")
    if st.button("Run Image Analysis (Doctor)"):
        st.info("Image analysis is not integrated yet.")
        if uploaded_file is not None and 'tmp_path' in locals():
            st.write("Uploaded file local path (use for integration):")
            st.code(tmp_path)
        else:
            st.write("No CT upload detected. Use the developer-provided dataset image path:")
            st.code(DATASET_IMAGE_LOCAL_PATH)
        st.warning("Image analysis functionality will be added later. No model prediction is performed here.")

st.markdown("---")
st.caption("Notes: Place 'lung_cancer_model.pkl' in the same directory as app.py. FEATURE_ORDER must exactly match the order used when training the model. If training involved scaling/encoders, load and apply identical preprocessing before calling .predict().")
