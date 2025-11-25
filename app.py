# app.py
import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
from io import BytesIO
from PIL import Image

# Path to the uploaded image (from your conversation history / dev message)
DATASET_IMAGE_PATH = "/mnt/data/7fff4457-81ee-4946-abe9-827fd7ca287b.png"

st.set_page_config(page_title="Lung Cancer Predictor", layout="wide")

st.title("Lung Cancer Prediction — Doctor / Patient Modes")

st.sidebar.header("Mode selection")
mode = st.sidebar.radio("Choose mode", ["Patient", "Doctor"])

st.sidebar.markdown("---")
st.sidebar.markdown("Model source (GitHub raw URL):")
GITHUB_RAW_URL = st.sidebar.text_input(
    "Enter raw URL to .pkl model file",
    value="", # put your raw GitHub URL here, e.g. https://raw.githubusercontent.com/username/repo/branch/path/model.pkl
)

# Helper to load model from GitHub raw URL
@st.cache_data(show_spinner=False)
def load_model_from_url(raw_url: str):
    if not raw_url:
        return None, "No URL provided"
    try:
        resp = requests.get(raw_url, timeout=10)
        resp.raise_for_status()
        model = joblib.load(BytesIO(resp.content))
        return model, None
    except Exception as e:
        return None, str(e)

# Display dataset image (if exists)
try:
    img = Image.open(DATASET_IMAGE_PATH)
    st.sidebar.image(img, caption="Dataset preview", use_column_width=True)
except Exception:
    pass

# Define expected features order for the model input.
# IMPORTANT: Adjust this list if your model expects a different feature order or different names.
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
    "Anxiety + yellow fingers"  # derived column
]

st.write("Feature order used for prediction (ensure this matches the model's training order):")
st.write(FEATURE_ORDER)

# Build input UI depending on mode
def boolean_input(label, key, default=0):
    # Streamlit selectbox for 0/1 to be explicit
    return int(st.number_input(label, min_value=0, max_value=1, value=default, step=1, key=key))

if mode == "Patient":
    st.header("Patient mode — enter symptoms (cancer NOT requested as input)")
    st.info("You provide symptoms below. The combined feature 'Anxiety + yellow fingers' is calculated automatically from ANXIETY and YELLOW_FINGERS.")
    with st.form("patient_form"):
        inputs = {}
        # Take inputs except cancer
        inputs["SMOKING"] = boolean_input("SMOKING (0/1)", "p_smoking", 0)
        inputs["YELLOW_FINGERS"] = boolean_input("YELLOW_FINGERS (0/1)", "p_yellow", 0)
        inputs["ANXIETY"] = boolean_input("ANXIETY (0/1)", "p_anxiety", 0)
        inputs["PEER_PRESSURE"] = boolean_input("PEER_PRESSURE (0/1)", "p_peer", 0)
        inputs["CHRONIC_DISEASE"] = boolean_input("CHRONIC_DISEASE (0/1)", "p_chronic", 0)
        inputs["FATIGUE"] = boolean_input("FATIGUE (0/1)", "p_fatigue", 0)
        inputs["ALLERGY"] = boolean_input("ALLERGY (0/1)", "p_allergy", 0)
        inputs["WHEEZING"] = boolean_input("WHEEZING (0/1)", "p_wheezing", 0)
        inputs["ALCOHOL_CONSUMING"] = boolean_input("ALCOHOL_CONSUMING (0/1)", "p_alcohol", 0)
        inputs["COUGHING"] = boolean_input("COUGHING (0/1)", "p_coughing", 0)
        inputs["SHORTNESS_OF_BREATH"] = boolean_input("SHORTNESS_OF_BREATH (0/1)", "p_shortness", 0)
        inputs["SWALLOWING_DIFFICULTY"] = boolean_input("SWALLOWING_DIFFICULTY (0/1)", "p_swallow", 0)
        inputs["CHEST_PAIN"] = boolean_input("CHEST_PAIN (0/1)", "p_chest", 0)

        submitted = st.form_submit_button("Predict (Patient)")

    if submitted:
        # Compute derived feature: "Anxiety + yellow fingers"
        # Current rule: logical OR => 1 if either anxiety or yellow_fingers is 1.
        # If you want a different rule (sum or AND), change this line accordingly.
        derived = 1 if (inputs["ANXIETY"] == 1 or inputs["YELLOW_FINGERS"] == 1) else 0
        inputs["Anxiety + yellow fingers"] = derived

        # Arrange features in FEATURE_ORDER
        try:
            x = np.array([inputs[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
        except KeyError as e:
            st.error(f"Feature mismatch: missing {e}. Check FEATURE_ORDER vs provided inputs.")
            st.stop()

        model, err = load_model_from_url(GITHUB_RAW_URL)
        if model is None:
            st.error(f"Model load failed: {err}")
        else:
            try:
                pred = model.predict(x)
                # If model returns array, take first element
                if hasattr(pred, "__len__"):
                    pred_value = pred[0]
                else:
                    pred_value = pred
                st.success(f"Predicted value (model output): **{pred_value}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif mode == "Doctor":
    st.header("Doctor mode — full inputs (you can also provide LUNG_CANCER value if desired)")
    st.info("Doctor mode allows editing all features including the LUNG_CANCER column. Prediction still uses the model loaded from the GitHub .pkl.")
    with st.form("doctor_form"):
        d_inputs = {}
        d_inputs["SMOKING"] = boolean_input("SMOKING (0/1)", "d_smoking", 0)
        d_inputs["YELLOW_FINGERS"] = boolean_input("YELLOW_FINGERS (0/1)", "d_yellow", 0)
        d_inputs["ANXIETY"] = boolean_input("ANXIETY (0/1)", "d_anxiety", 0)
        d_inputs["PEER_PRESSURE"] = boolean_input("PEER_PRESSURE (0/1)", "d_peer", 0)
        d_inputs["CHRONIC_DISEASE"] = boolean_input("CHRONIC_DISEASE (0/1)", "d_chronic", 0)
        d_inputs["FATIGUE"] = boolean_input("FATIGUE (0/1)", "d_fatigue", 0)
        d_inputs["ALLERGY"] = boolean_input("ALLERGY (0/1)", "d_allergy", 0)
        d_inputs["WHEEZING"] = boolean_input("WHEEZING (0/1)", "d_wheezing", 0)
        d_inputs["ALCOHOL_CONSUMING"] = boolean_input("ALCOHOL_CONSUMING (0/1)", "d_alcohol", 0)
        d_inputs["COUGHING"] = boolean_input("COUGHING (0/1)", "d_coughing", 0)
        d_inputs["SHORTNESS_OF_BREATH"] = boolean_input("SHORTNESS_OF_BREATH (0/1)", "d_shortness", 0)
        d_inputs["SWALLOWING_DIFFICULTY"] = boolean_input("SWALLOWING_DIFFICULTY (0/1)", "d_swallow", 0)
        d_inputs["CHEST_PAIN"] = boolean_input("CHEST_PAIN (0/1)", "d_chest", 0)

        # This is the cancer column (doctor may input)
        lung_cancer_input = st.selectbox("LUNG_CANCER (0/1) - only for record, model prediction is still independent",
                                         options=[0, 1], index=0, key="d_cancer")

        submit_doc = st.form_submit_button("Predict (Doctor)")

    if submit_doc:
        derived = 1 if (d_inputs["ANXIETY"] == 1 or d_inputs["YELLOW_FINGERS"] == 1) else 0
        d_inputs["Anxiety + yellow fingers"] = derived

        try:
            xdoc = np.array([d_inputs[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
        except KeyError as e:
            st.error(f"Feature mismatch: missing {e}. Check FEATURE_ORDER vs provided inputs.")
            st.stop()

        model, err = load_model_from_url(GITHUB_RAW_URL)
        if model is None:
            st.error(f"Model load failed: {err}")
        else:
            try:
                pred = model.predict(xdoc)
                if hasattr(pred, "__len__"):
                    pred_value = pred[0]
                else:
                    pred_value = pred
                st.success(f"Predicted value (model output): **{pred_value}**")
                st.write("Note: You entered LUNG_CANCER value (doctor entry):", lung_cancer_input)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Footer notes
st.markdown("---")
st.caption("Notes: \n- Ensure the model's expected feature order and preprocessing match FEATURE_ORDER above. \n- If the model used scaling or other preprocessing at training time, apply identical preprocessing before prediction.")
