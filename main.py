import os
import librosa
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import joblib

from utils.feature_extraction import extract_features

DATA_DIR = 'audio_files/'  

def get_features_and_labels():
    features = []
    labels = []
    
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                data, sr = librosa.load(filepath, duration=3, offset=0.5)
                feat = extract_features(filepath)
                label = file.split("-")[2]  
                features.append(feat)
                labels.append(label)
    
    return np.array(features), np.array(labels)

def train_model():
    X, y = get_features_and_labels()

    encoder = LabelEncoder()
    y_encoded = to_categorical(encoder.fit_transform(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(y_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

    model.save('model/audio_sentiment_model.h5')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(encoder, 'model/label_encoder.pkl')

    print("Model training complete and saved.")

if __name__ == "__main__":
    train_model()
