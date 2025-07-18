import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import tempfile
from utils.featureExtraction import extract_features  # This version accepts file_path

# Load model and preprocessing tools
model = tf.keras.models.load_model("model/audio_sentiment_model.keras")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Page configuration
st.set_page_config(page_title="Audio Emotion Recognition", layout="centered")
st.title("🎙️ Audio Emotion Recognition")
st.markdown("Upload a short **WAV audio clip** and the model will predict the **emotion** being expressed.")

# File uploader
uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if uploaded_file is not None:
    try:
        # Play uploaded audio
        st.audio(uploaded_file, format='audio/wav')

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Extract features using file path
        features = extract_features(temp_path)

        if features is None or np.isnan(features).any():
            st.error("Failed to extract valid features. Please upload a clearer or slightly longer audio clip.")
        else:
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            # Display prediction
            st.subheader(" Predicted Emotion:")
            st.success(f"**{predicted_label}**")

            # Show probabilities
            st.subheader(" Prediction Probabilities:")
            probs = {label: f"{prob*100:.2f}%" for label, prob in zip(label_encoder.classes_, prediction[0])}
            st.json(probs)

    except Exception as e:
        st.error(f" Error: {e}")
