import os
import librosa
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from glob import glob

def extract_mel(path, sr=22050, n_mels=128, duration=4):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    if len(y) < sr*duration:
        y = np.pad(y, (0, sr*duration - len(y)))
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_db

X = []
y = []
classes = sorted(os.listdir("audio_dataset"))
class_to_idx = {c:i for i,c in enumerate(classes)}

for cls in classes:
    for f in glob(os.path.join("audio_dataset", cls, "*.wav")):
        mel = extract_mel(f)
        X.append(mel)
        y.append(class_to_idx[cls])

X = np.array(X)[..., np.newaxis]  # shape (n, n_mels, time, 1)
y = to_categorical(y, num_classes=len(classes))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=40, batch_size=16)

model.save("models/audio_model.h5")
# save classes mapping
import json
with open("models/audio_classes.json","w") as f:
    json.dump(class_to_idx, f)