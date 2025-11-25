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
    """ Interpret model output and present a user-friendly message.
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
        "Psych / Other": ["Anxiety"],
    }

    with st.form("patient_form"):
        selected_labels = []

        cols = st.columns(2)
        for i, (group_name, labels) in enumerate(GROUPS.items()):
            col = cols[i % 2]
            col.markdown(f"### {group_name}")

            chosen = col.multiselect(
                label="",
                options=labels,
                default=[]
            )

            for ch in chosen:
                if ch not in selected_labels:
                    selected_labels.append(ch)

        submitted = st.form_submit_button("Predict")
        if submitted:
            inputs = {}
            for label, key in FEATURE_LABEL_MAP.items():
                inputs[key] = 1 if label in selected_labels else 0

            inputs["Anxiety + yellow fingers"] = 1 if (
                inputs["ANXIETY"] == 1 or inputs["YELLOW_FINGERS"] == 1
            ) else 0

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

# ------------------- DOCTOR MODE (complete block) -------------------
# Doctor Mode — complete block (drop-in replacement)
# - CSV doctor login (doctors.csv at repo root or upload via UI)
# - Upload CT image (or use developer-provided fallback image)
# - On "Run Image Analysis" the app POSTS a JSON payload {"file_url": <local_path>} to a remote inference endpoint
#   (so nothing heavy runs locally). If no endpoint is configured, the app displays the local file path so your tooling
#   can transform it to a public URL and call the inference service externally.
# - No TensorFlow is imported / used here.
#
# Important:
# - This block expects to be placed inside your existing app.py where Doctor Mode code belongs.
# - The file path below is the file uploaded in this session and will be sent as the "file_url":
#     /mnt/data/bc397ec8-5d3e-4aa5-a0a2-533368b0dc44.png

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from PIL import Image
import requests
import json
import time

# ---------------- CONFIG / PATHS ----------------
# Local image file that was uploaded in this session (developer-provided path)
SESSION_FILE_URL = "/mnt/data/bc397ec8-5d3e-4aa5-a0a2-533368b0dc44.png"

# doctors.csv expected at repo root
DOCTORS_CSV_PATH = Path("doctors.csv")

# By default we won't show or require any model URL in the UI.
# If you have a remote inference endpoint, set it in Streamlit Secrets:
# st.secrets["INFERENCE_URL"] = "https://your-inference-server.example/predict"
# OR you may paste it in the hidden sidebar field below (not shown by default).
#
# The app will send a JSON payload: {"file_url": "<local_path_here>"} to that endpoint.
# Your external tooling is expected to transform the local path into a reachable URL when receiving it,
# or you can host the file and provide an endpoint that accepts the raw file bytes as multipart/form-data.
#
# ---------------- Helpers ----------------

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

def post_file_url_to_inference(endpoint: str, file_url: str, timeout: int = 60):
    """
    POSTs a JSON payload {"file_url": file_url} to the inference endpoint.
    Returns (response_json, error_message).
    """
    headers = {"Content-Type": "application/json"}
    payload = {"file_url": file_url}
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        # Expect JSON like {"prediction": 0.12} or {"prediction": 1} etc.
        return resp.json(), None
    except Exception as e:
        return None, str(e)

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
            st.info("You may upload a doctors.csv below to enable login.")
        else:
            st.session_state["doctors_dict"] = docs
    else:
        st.info("No doctors.csv found. You can upload one below to enable login.")

# Upload doctors.csv (to create or replace)
with st.expander("Upload / Replace doctors.csv (CSV columns: doctor_id,password)", expanded=False):
    uploaded_docs = st.file_uploader("Upload doctors.csv", type=["csv"], key="doctors_csv_upload_doctor")
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
    with st.form("doctor_login_form_ui"):
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

    # Clean UI: no blue info bar, no Dropbox/model link shown to user
    st.header("CT scan upload & analysis")
    st.write("Upload a CT image (DICOM/JPEG/PNG). Click **Run Image Analysis** to request remote inference. Nothing heavy runs locally.")

    # File uploader (doctor can still upload a CT)
    uploaded_file = st.file_uploader("Upload CT scan image (DICOM/JPEG/PNG)", type=["png", "jpg", "jpeg", "dcm"], key="doctor_ct_upload_ui")

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
                st.info("Preview not available (file may be DICOM or unsupported).")
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
    else:
        st.info("No CT uploaded yet. You may upload or use the developer dataset preview below.")

    # Show developer-provided dataset preview (plain text + image). This is the local path in session history.
    st.subheader("Dataset preview (developer-provided)")
    try:
        if os.path.exists(SESSION_FILE_URL):
            preview_img = Image.open(SESSION_FILE_URL)
            st.image(preview_img, caption="Developer-provided preview", use_column_width=True)
            st.write("Developer-provided local file path (will be sent as file_url to inference service):")
            st.code(SESSION_FILE_URL)
        else:
            st.warning("Developer-provided preview file not found on disk.")
    except Exception:
        st.warning("Unable to open developer-provided preview image.")

    st.markdown("---")
    # Hidden: remote inference endpoint can be configured via secrets or (optionally) a hidden input
    inference_url_from_secrets = st.secrets.get("INFERENCE_URL", "") if hasattr(st, "secrets") else ""
    # Optionally allow admin to paste an endpoint in a collapsed expander (not shown by default)
    with st.expander("(Optional) Set remote inference endpoint (advanced)", expanded=False):
        inference_url_override = st.text_input("Inference endpoint URL (e.g. https://server.example/predict)", value=inference_url_from_secrets or "")
        st.caption("If set, the app will POST {\"file_url\": \"<local_path>\"} to this endpoint.")

    # Decide which endpoint to use (priority: secrets -> expander override -> none)
    inference_endpoint = inference_url_from_secrets or (inference_url_override.strip() if inference_url_override else "")

    # When Run Image Analysis is clicked, we will either:
    # - POST {"file_url": <local_path>} to the configured inference endpoint (so nothing heavy runs locally), OR
    # - If no endpoint is configured, show the local path and instructions so your toolchain can transform the path into a reachable URL and call the inference.
    if st.button("Run Image Analysis"):
        # choose image path to "send"
        image_path_to_send = uploaded_local_path if uploaded_local_path else SESSION_FILE_URL

        if not os.path.exists(image_path_to_send):
            st.error("No image available to analyze. Upload a CT scan or ensure the developer dataset file exists on the session filesystem.")
        else:
            # If an inference endpoint is configured, POST the local path (as JSON) to it.
            if inference_endpoint:
                st.info("Sending request to remote inference endpoint...")
                with st.spinner("Contacting inference server..."):
                    resp_json, err = post_file_url_to_inference(inference_endpoint, image_path_to_send)
                if err:
                    st.error(f"Failed to contact inference server: {err}")
                    st.info("Make sure the endpoint accepts JSON {\"file_url\": \"<local_path>\"} and that your external tooling can transform the local path into a reachable URL or can access the shared filesystem.")
                else:
                    # Interpret response JSON
                    # Expected response schema (example): {"prediction": 0.12} or {"prediction": 1}
                    pred_val = None
                    if isinstance(resp_json, dict):
                        # common key name
                        for key in ("prediction", "pred", "result", "score"):
                            if key in resp_json:
                                try:
                                    pred_val = float(resp_json[key])
                                except Exception:
                                    pred_val = None
                                break
                    # If prediction found, show Low/High (no probability)
                    if pred_val is None:
                        st.error("Inference server returned unexpected response. See raw response below:")
                        st.json(resp_json)
                    else:
                        if pred_val >= 0.5:
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
            else:
                # No endpoint configured: show the local path so your external tooling can transform it into a URL
                st.info("No remote inference endpoint configured.")
                st.write("We will NOT run any heavy model locally. Instead, please transform the following local path into a reachable URL from your inference tooling and call your model with it:")
                st.code(image_path_to_send)
                st.markdown(
                    """
                    **Suggested next steps for your integration tooling**
                    1. Make the local path accessible to your inference system (e.g., upload the file to a shared location or accept the path from the host).  
                    2. Call your model inference endpoint with JSON `{\"file_url\": \"<transformed_url>\"}` or send the raw image bytes as multipart/form-data.  
                    3. Return JSON like `{\"prediction\": 0.12}` and the Streamlit client will interpret it as Low/High using a 0.5 threshold.
                    """
                )
                st.caption(f"Local file path provided: {image_path_to_send}")

    # small audit log UI (optional): allow doctor to save a record locally (append)
    with st.expander("Optional: save local audit record of this action", expanded=False):
        if st.button("Append audit record"):
            audit_file = Path("doctor_audit_log.csv")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            entry = {"doctor_id": st.session_state.get("doctor_id", ""), "file_path": (uploaded_local_path or SESSION_FILE_URL), "timestamp": timestamp}
            # append as CSV row
            header = not audit_file.exists()
            try:
                import csv
                with open(audit_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["doctor_id", "file_path", "timestamp"])
                    if header:
                        writer.writeheader()
                    writer.writerow(entry)
                st.success(f"Appended audit record to {audit_file.resolve()}")
            except Exception as e:
                st.error(f"Failed to append audit record: {e}")
