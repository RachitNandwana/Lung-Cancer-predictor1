import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import requests
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------------------------------------
#                MAIN SIDEBAR MODE SELECTOR
# -----------------------------------------------------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode", ["Patient", "Doctor"])


# ===========================================================
#                       PATIENT MODE
# ===========================================================
if mode == "Patient":
    st.title("🧍 Patient Mode – Lung Cancer Risk Checker")

    st.write("Answer the following questions to help assess risk.")

    name = st.text_input("Enter your name")
    age = st.number_input("Age", 1, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
    alcohol = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
    cough = st.selectbox("Do you have persistent cough?", ["Yes", "No"])
    chest_pain = st.selectbox("Do you feel chest pain?", ["Yes", "No"])
    weight_loss = st.selectbox("Unusual weight loss?", ["Yes", "No"])

    submit = st.button("Submit")

    if submit:
        st.success("Form submitted successfully! A doctor will review your responses.")
        st.json({
            "name": name,
            "age": age,
            "gender": gender,
            "smoking": smoking,
            "alcohol": alcohol,
            "cough": cough,
            "chest pain": chest_pain,
            "weight loss": weight_loss
        })



# ===========================================================
#                        DOCTOR MODE
# ===========================================================
elif mode == "Doctor":

    st.title("🩺 Doctor Mode – Login Required")

    # -------------------------------------------------------
    #                CONFIG & PATHS
    # -------------------------------------------------------
    MODEL_URL = "https://www.dropbox.com/scl/fi/bab21h5uesf59j2yn9ngu/lung_cancer_model_finalmodel.h5?rlkey=uw8n849or6sfcolvevbjngahh&st=cp4zn6qy&dl=0"
    DOCTORS_CSV_PATH = Path("doctors.csv")

    # -------------------------------------------------------
    #        LOAD DOCTOR CREDENTIALS FROM CSV
    # -------------------------------------------------------
    def load_doctors_from_csv(csv_path: Path):
        try:
            df = pd.read_csv(csv_path)
            if "doctor_id" not in df.columns or "password" not in df.columns:
                return None
            return dict(zip(df["doctor_id"], df["password"]))
        except Exception:
            return None

    # -------------------------------------------------------
    #      UTIL: Convert Dropbox URL → Direct Download URL
    # -------------------------------------------------------
    def dropbox_to_direct(url: str) -> str:
        if "dl=0" in url:
            return url.replace("dl=0", "dl=1")
        return url

    # -------------------------------------------------------
    #      UTIL: Download Model File
    # -------------------------------------------------------
    def download_large_file(url: str):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            with open(tmp.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return tmp.name
        except:
            return None

    # -------------------------------------------------------
    #      UTIL: Preprocess Image for Model
    # -------------------------------------------------------
    def preprocess_image_for_model(image):
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        return image


    # -------------------------------------------------------
    #                  SESSION STATE
    # -------------------------------------------------------
    if "doctor_logged_in" not in st.session_state:
        st.session_state["doctor_logged_in"] = False
    if "doctor_id" not in st.session_state:
        st.session_state["doctor_id"] = None

    if "doctors_dict" not in st.session_state:
        st.session_state["doctors_dict"] = load_doctors_from_csv(DOCTORS_CSV_PATH)


    # -------------------------------------------------------
    #                     LOGIN SCREEN
    # -------------------------------------------------------
    if not st.session_state["doctor_logged_in"]:

        st.subheader("Doctor Login")

        with st.form("login_form"):
            doc_id = st.text_input("Doctor ID")
            doc_pw = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")

        if login:
            creds = st.session_state["doctors_dict"]
            if creds and doc_id in creds and creds[doc_id] == doc_pw:
                st.session_state["doctor_logged_in"] = True
                st.session_state["doctor_id"] = doc_id
                st.success("Login successful! Redirecting...")
                st.experimental_rerun()
            else:
                st.error("Invalid ID or password.")

        st.stop()


    # -------------------------------------------------------
    #                LOGGED-IN DOCTOR DASHBOARD
    # -------------------------------------------------------
    st.success(f"Logged in as: **{st.session_state['doctor_id']}**")

    if st.button("Logout"):
        st.session_state["doctor_logged_in"] = False
        st.experimental_rerun()

    st.header("📤 Upload CT Scan for Analysis")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])


    # -------------------------------------------------------
    #               MODEL LOADING
    # -------------------------------------------------------
    st.write("📦 Loading AI Model...")

    direct_link = dropbox_to_direct(MODEL_URL)
    model_path = download_large_file(direct_link)

    if model_path:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load model.")
        st.stop()


    # -------------------------------------------------------
    #                 PREDICTION SECTION
    # -------------------------------------------------------
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded CT Scan", use_container_width=True)

        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        processed = preprocess_image_for_model(img_np)

        pred = model.predict(processed)
        result = pred[0][0]

        st.subheader("🔍 Prediction Result")

        if result > 0.5:
            st.error(f"⚠️ High chance of lung cancer (Confidence: {result:.2f})")
        else:
            st.success(f"👍 Low chance of lung cancer (Confidence: {result:.2f})")
