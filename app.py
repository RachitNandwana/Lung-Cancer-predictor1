# app.py
import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import tempfile

st.set_page_config(page_title="Lung Cancer Predictor", layout="wide")
st.title("Lung Cancer Prediction — Doctor / Patient Modes")

# Developer-provided dataset image local path (use as "URL" for integration step)
DATASET_IMAGE_LOCAL_PATH = "/mnt/data/7fff4457-81ee-4946-abe9-827fd7ca287b.png"

# Local model filename expected in same repo as app.py
LOCAL_MODEL_PATH = Path("lung_cancer_model.pkl")

st.sidebar.header("Mode selection")
mode = st.sidebar.radio("Choose mode", ["Patient", "Doctor"])

st.markdown("---")

# Feature order the model expects (must match training)
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

st.caption("Ensure FEATURE_ORDER matches the model's training order and preprocessing.")

# ---------- helpers ----------
def binary_input(label, key, default=0):
    return int(st.number_input(label, min_value=0, max_value=1, value=default, step=1, key=key))

def compute_derived(anx, yellow):
    # Rule: logical OR -> 1 if either is 1. Change if your training used another rule.
    return 1 if (anx == 1 or yellow == 1) else 0

def load_local_model(path: Path):
    if not path.exists():
        return None, f"Model file not found at: {path.resolve()}"
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

def show_prediction_only_and_warning(model, x):
    try:
        pred = model.predict(x)
        # extract scalar prediction
        pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        st.success(f"Predicted value (model output): **{pred_value}**")
        st.warning("**Disclaimer:** Automated prediction may vary. This is not a medical diagnosis. Consult a qualified healthcare professional.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------- Patient Mode ----------
# ---------- Patient Mode ----------
if mode == "Patient":
    st.header("Patient Mode")

    with st.form("patient_form"):
        st.subheader("Enter your symptoms")

        def yn(label, key):
            return 1 if st.radio(label, ["No", "Yes"], horizontal=True, key=key) == "Yes" else 0

        inputs = {}
        inputs["SMOKING"] = yn("Smoking", "p_smoking")
        inputs["YELLOW_FINGERS"] = yn("Yellow Fingers", "p_yellow")
        inputs["ANXIETY"] = yn("Anxiety", "p_anxiety")
        inputs["PEER_PRESSURE"] = yn("Peer Pressure", "p_peer")
        inputs["CHRONIC_DISEASE"] = yn("Chronic Disease", "p_chronic")
        inputs["FATIGUE"] = yn("Fatigue", "p_fatigue")
        inputs["ALLERGY"] = yn("Allergy", "p_allergy")
        inputs["WHEEZING"] = yn("Wheezing", "p_wheezing")
        inputs["ALCOHOL_CONSUMING"] = yn("Alcohol Consumption", "p_alcohol")
        inputs["COUGHING"] = yn("Coughing", "p_coughing")
        inputs["SHORTNESS_OF_BREATH"] = yn("Shortness of Breath", "p_shortness")
        inputs["SWALLOWING_DIFFICULTY"] = yn("Swallowing Difficulty", "p_swallow")
        inputs["CHEST_PAIN"] = yn("Chest Pain", "p_chest")

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Auto compute derived column
        inputs["Anxiety + yellow fingers"] = 1 if (
            inputs["ANXIETY"] == 1 or inputs["YELLOW_FINGERS"] == 1
        ) else 0

        x = np.array([inputs[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)

        model, err = load_local_model(LOCAL_MODEL_PATH)
        if model is None:
            st.error(f"Model load failed: {err}")
        else:
            show_prediction_only_and_warning(model, x)

# ---------- Doctor Mode ----------
else:
    st.header("Doctor mode — CT scan / image analysis (not integrated)")
    st.info("Doctor mode currently supports uploading a CT scan image for future analysis. Image analysis is not integrated yet; click 'Run Image Analysis' to get the file path for integration.")

    # Allow doctor to upload their own CT scan (or use the dataset preview image)
    uploaded_file = st.file_uploader("Upload CT scan image (DICOM/JPEG/PNG) for future analysis", type=["png", "jpg", "jpeg", "dcm"])

    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file is not None:
            st.success("Uploaded file received.")
            # Save uploaded to a temp file and present the local path (so your tooling can transform it into a URL)
            try:
                suffix = Path(uploaded_file.name).suffix or ".img"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_file.getbuffer())
                tmp.flush()
                tmp_path = tmp.name
                tmp.close()
                st.write("Temporary local file path (use this as file URL for integration):")
                st.code(tmp_path)
                # preview image if PIL-readable
                try:
                    im = Image.open(tmp_path)
                    st.image(im, caption="Uploaded image preview")
                except Exception:
                    st.info("Uploaded file preview not available (maybe DICOM or unsupported for preview).")
            except Exception as e:
                st.error(f"Failed to save uploaded file: {e}")
        else:
            st.info("No file uploaded. You may also use the dataset preview image (right).")

    with col2:
        st.subheader("Dataset preview image (developer-provided)")
        try:
            img = Image.open(DATASET_IMAGE_LOCAL_PATH)
            st.image(img, caption="Dataset preview")
            st.write("Local dataset image path (use this as file URL for integration):")
            st.code(DATASET_IMAGE_LOCAL_PATH)
        except Exception:
            st.warning("Dataset preview image not available at the expected path.")

    st.markdown("---")
    if st.button("Run Image Analysis (Doctor)"):
        # Show message that image analysis isn't integrated yet
        st.info("Image analysis is not integrated yet.")
        # If a user uploaded, show that temp path; otherwise show developer-provided dataset path
        if uploaded_file is not None and 'tmp_path' in locals():
            st.write("Use this local file path for integration (we will transform it into a URL in the tool call):")
            st.code(tmp_path)
        else:
            st.write("No CT upload detected. Use the developer-provided dataset image local path for integration:")
            st.code(DATASET_IMAGE_LOCAL_PATH)
        st.warning("Image analysis functionality will be added later. No model prediction is performed here.")

st.markdown("---")
st.caption("Notes: \n- This app expects a local model file named 'lung_cancer_model.pkl' in the same directory as app.py. \n- FEATURE_ORDER must exactly match the feature order used when training the model. \n- If your training pipeline applied scaling/encoders, load and apply the same preprocessing objects before predicting.")
