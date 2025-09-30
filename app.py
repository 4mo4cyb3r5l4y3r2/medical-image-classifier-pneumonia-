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

# --- MODEL AND IMAGE CONSTANTS ---
IMAGE_SIZE = (224, 224)
# **IMPORTANT**: Paste the direct download link to your .keras file from GitHub Releases
MODEL_URL = "https://github.com/4mo4cyb3r5l4y3r2/medical-image-classifier-pneumonia-/releases/download/v1.0/best_pneumonia_finetuned.keras" 
MODEL_FILENAME = "best_pneumonia_finetuned.keras"

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_pneumonia_model():
    """Downloads the model from the URL and loads it into memory."""
    try:
        model_path = tf.keras.utils.get_file(
            fname=MODEL_FILENAME,
            origin=MODEL_URL
        )
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_pneumonia_model()

# --- WEB APP INTERFACE ---
st.title("Chest X-Ray Pneumonia Detection 🩺")
st.write("Upload a chest X-ray image, and a fine-tuned ResNet50 model will predict if it shows signs of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None and model is not None:
    st.image(uploaded_file, caption='Uploaded X-Ray.', use_column_width=True)
    st.write("")
    
    with st.spinner('Classifying...'):
        try:
            # Pre-process the image
            img = image.load_img(uploaded_file, target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            prediction = model.predict(img_array)
            probability = prediction[0][0]

            # Display the result
            st.write("--- Prediction Result ---")
            if probability > 0.5:
                st.error(f"Prediction: PNEUMONIA (Confidence: {probability * 100:.2f}%)")
            else:
                st.success(f"Prediction: NORMAL (Confidence: {(1 - probability) * 100:.2f}%)")
                
            st.info("Disclaimer: This is an educational project and not a substitute for a professional medical diagnosis.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")