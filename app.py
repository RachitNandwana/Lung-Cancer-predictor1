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
import pandas as pd

DOCTORS_CSV_PATH = Path("doctors.csv")  # expected CSV at repo root
DEV_DATASET_IMAGE_PATH = "/mnt/data/7fff4457-81ee-4946-abe9-827fd7ca287b.png"

def load_doctors_from_csv(path: Path):
    """Return dict {doctor_id: password} or (None, err)"""
    if not path.exists():
        return None, f"doctors CSV not found at {path.resolve()}"
    try:
        df = pd.read_csv(path)
        # Expecting columns 'doctor_id' and 'password'
        if "doctor_id" not in df.columns or "password" not in df.columns:
            return None, "CSV must contain 'doctor_id' and 'password' columns"
        return dict(zip(df["doctor_id"].astype(str), df["password"].astype(str))), None
    except Exception as e:
        return None, str(e)

# Initialize session state for doctor login
if "doctor_logged_in" not in st.session_state:
    st.session_state["doctor_logged_in"] = False
if "doctor_id" not in st.session_state:
    st.session_state["doctor_id"] = None
if "doctors_dict" not in st.session_state:
    st.session_state["doctors_dict"] = None  # will hold loaded credentials

st.header("Doctor Mode — login required")

# If CSV exists, attempt to load it into session_state (so we don't re-read every render)
if st.session_state["doctors_dict"] is None:
    if DOCTORS_CSV_PATH.exists():
        docs, err = load_doctors_from_csv(DOCTORS_CSV_PATH)
        if docs is None:
            st.error(f"Failed to load doctors.csv: {err}")
            st.info("You can upload a replacement CSV below.")
        else:
            st.session_state["doctors_dict"] = docs
    else:
        st.info("No doctors.csv found in repository. You can upload a doctors CSV now to enable login.")

# Allow uploading a doctors CSV (to create or replace doctors.csv)
with st.expander("Upload / Replace doctors.csv (CSV columns: doctor_id,password)", expanded=False):
    uploaded_docs = st.file_uploader("Upload doctors.csv", type=["csv"], key="doctors_csv_upload")
    if uploaded_docs is not None:
        try:
            df_new = pd.read_csv(uploaded_docs)
            if "doctor_id" not in df_new.columns or "password" not in df_new.columns:
                st.error("CSV must contain 'doctor_id' and 'password' columns.")
            else:
                # Save to repo root (works in local/dev). On Streamlit Cloud, this will persist per deploy.
                df_new.to_csv(DOCTORS_CSV_PATH, index=False)
                st.success(f"Saved doctors.csv to {DOCTORS_CSV_PATH.resolve()}. Reloading credentials...")
                docs, err = load_doctors_from_csv(DOCTORS_CSV_PATH)
                if docs is not None:
                    st.session_state["doctors_dict"] = docs
                else:
                    st.error(f"Failed to load uploaded CSV: {err}")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# If not logged in, show login form
if not st.session_state["doctor_logged_in"]:
    with st.form("doctor_login_form"):
        doc_id = st.text_input("Doctor ID")
        doc_pw = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        if st.session_state["doctors_dict"] is None:
            st.error("No credentials available. Upload a doctors.csv or add credentials to doctors.csv in repo root.")
        else:
            # validate
            creds = st.session_state["doctors_dict"]
            if doc_id in creds and creds[doc_id] == doc_pw:
                st.session_state["doctor_logged_in"] = True
                st.session_state["doctor_id"] = doc_id
                st.success(f"Logged in as {doc_id}")
                st.experimental_rerun()
            else:
                st.error("Invalid Doctor ID or password.")

# If logged in, show doctor dashboard (CT upload area + dataset preview)
else:
    st.markdown(f"**Logged in as:** `{st.session_state['doctor_id']}`")
    if st.button("Logout"):
        st.session_state["doctor_logged_in"] = False
        st.session_state["doctor_id"] = None
        st.experimental_rerun()

    st.subheader("CT scan upload (image analysis not integrated yet)")
    uploaded_file = st.file_uploader("Upload CT scan image (DICOM/JPEG/PNG)", type=["png", "jpg", "jpeg", "dcm"], key="doctor_ct_upload")

    # show upload preview and save temp file if provided
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
            try:
                img = Image.open(tmp_path)
                st.image(img, caption="Uploaded CT preview", use_column_width=True)
            except Exception:
                st.info("Preview not available (possibly DICOM).")
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
    else:
        st.info("No CT uploaded yet. You can upload or use the dataset preview below.")

    st.markdown("### Dataset preview (developer-provided)")
    try:
        img = Image.open(DEV_DATASET_IMAGE_PATH)
        st.image(img, caption="Dataset preview", use_column_width=True)
    except Exception:
        st.warning("Dataset preview image not available at expected path.")

    st.write("File URL to use for integration (developer-provided):")
    st.code(DEV_DATASET_IMAGE_PATH)
    st.info("Image analysis is not integrated yet. When ready, call your image-analysis endpoint with the uploaded file path or this dataset path.")
