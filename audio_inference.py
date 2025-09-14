import librosa
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import os
import soundfile as sf

audio_model = tf.keras.models.load_model("models/audio_model.h5")
with open("models/audio_classes.json","r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v:k for k,v in class_to_idx.items()}

def prepare_mel_for_inference(file_stream, sr=22050, duration=4, n_mels=128):
    # file_stream: bytes or file path
    # read via soundfile if bytes
    y, _ = sf.read(file_stream, dtype='float32')
    # if stereo -> mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # ensure duration
    if len(y) < sr*duration:
        y = np.pad(y, (0, sr*duration - len(y)))
    else:
        y = y[:sr*duration]
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_db[np.newaxis,...,np.newaxis]

def run_audio_detection(file_path):
    X = prepare_mel_for_inference(file_path)
    pred = audio_model.predict(X)[0]
    idx = int(np.argmax(pred))
    label = idx_to_class[idx]
    confidence = float(pred[idx])
    # map to explanation
    AUDIO_EXPLANATIONS = {
        "normal": ("Normal motor sesi", "Önemli bir anormallik tespit edilmedi."),
        "knock": ("Tıkırtı / vuruntu", "Silindir/sıkıştırma veya ateşleme kontrolü önerilir."),
        "bearing_noise": ("Rulman sesi", "Rulman kontrolü/gerekiyorsa değişimi önerilir."),
        # ...
    }
    title, advice = AUDIO_EXPLANATIONS.get(label, (label, "Öneri yok"))
    return {"label": label, "confidence": confidence, "title": title, "advice": advice}