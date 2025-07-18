# 🎧 Audio Sentimental Analysis

Audio Sentimental Analysis is a speech-based emotion recognition system that detects human emotions such as **Happy**, **Sad**, **Angry**, **Neutral**, etc., using audio signals. It leverages deep learning and audio signal processing to classify emotions from `.wav` files.

---

## 🚀 Features

- Upload `.wav` audio clips (≤ 3 seconds)
- Emotion prediction using a trained neural network model
- Real-time prediction confidence scores
- Displays probability distribution for each emotion
- Handles invalid or low-quality audio inputs gracefully

---

## 🧠 Technologies Used

| Technology        | Purpose                         |
|------------------|---------------------------------|
| Python           | Core programming language       |
| Streamlit        | Web interface                   |
| Librosa          | Audio feature extraction        |
| TensorFlow/Keras | Deep learning model             |
| Scikit-learn     | Preprocessing and encoding      |
| Joblib           | Model and encoder persistence   |
| Pandas & NumPy   | Data manipulation               |

---

## 🗂️ Project Structure

audio-sentimental-analysis/
├── app.py # Streamlit web app
├── dataset/
│ └── features.csv # Extracted audio features
├── model/
│ ├── audio_sentiment_model.keras
│ ├── label_encoder.pkl
│ └── scaler.pkl
├── utils/
│ └── featureExtraction.py # Feature extraction logic
├── audioFiles/ # Training audio samples
├── extract_features_and_save.py
└── README.md

---

## 📈 Model Overview

- **Architecture**: Dense Neural Network
- **Layers**: 512 → 256 → 128 → 64 → Output (Softmax)
- **Techniques**: Dropout, BatchNormalization, EarlyStopping

---

## 🎼 Audio Features Used

- Zero Crossing Rate
- MFCCs (Mel-frequency cepstral coefficients)
- Chroma STFT
- RMS Energy
- Mel Spectrogram
- Spectral Contrast
- Spectral Bandwidth
- Spectral Rolloff

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/purvijoshi2/audio-sentimental-analysis.git
cd audio-sentimental-analysis
python -m venv env
env\Scripts\activate   # On Windows
pip install -r requirements.txt
streamlit run app.py
