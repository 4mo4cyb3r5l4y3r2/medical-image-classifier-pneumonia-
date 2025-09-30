import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩺",
    layout="wide"
)

# --- CONSTANTS ---
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'best_pneumonia_detector.keras'

# --- LOAD THE TRAINED MODEL ---
# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_pneumonia_model():
    """Loads and returns the trained Keras model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_pneumonia_model()

# --- WEB APP INTERFACE ---
st.title("Chest X-Ray Pneumonia Detection 🩺")
st.write("Upload a chest X-ray image, and the model will predict whether it shows signs of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded X-Ray.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # --- PRE-PROCESS THE IMAGE ---
        img = image.load_img(uploaded_file, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale

        # --- MAKE PREDICTION ---
        prediction = model.predict(img_array)
        probability = prediction[0][0]

        # --- DISPLAY THE RESULT ---
        st.write("--- Prediction Result ---")
        if probability > 0.5:
            st.error(f"Prediction: PNEUMONIA (Confidence: {probability * 100:.2f}%)")
        else:
            st.success(f"Prediction: NORMAL (Confidence: {(1 - probability) * 100:.2f}%)")
            
        st.info("Disclaimer: This is an educational project and not a substitute for a professional medical diagnosis.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")