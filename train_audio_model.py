# train_audio_model.py
import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Dataset klasör yapısı:
# audio_dataset/
#   normal/
#   kayis_gicirti/
#   motor_vuruntu/

DATASET_PATH = "audio_dataset"
X, y = [], []

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Veriyi yükle
for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(folder, file))
            X.append(features)
            y.append(label)

X, y = np.array(X), np.array(y)

# Eğitim/Test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Kaydet
joblib.dump(model, "audio_model.pkl")
print("✅ Audio model kaydedildi: audio_model.pkl")
