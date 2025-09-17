# audio_inference.py
# Bu dosya: ses dosyasından mel-spektrogram çıkarır, önceden eğitilmiş tf.keras modeline verir
# ve sesin sınıfını + güven skorunu + açıklama/advice döndürür.

import librosa                # ses işleme ve mel-spektrogram için
import numpy as np           # sayısal işlemler, dizi manipülasyonu
import tensorflow as tf      # model yükleme ve tahmin için (tf.keras)
import json                  # sınıf eşleştirme dosyasını okumak için
from datetime import datetime # (şu an kullanılmıyor ama zaman damgası gerekirse)
import os                    # dosya yolları ve filesystem işlemleri
import soundfile as sf       # ses dosyalarını okumak için (supports wav, flac, mp3 via libsndfile)

# Önceden eğitilmiş model dosyasını yükle.
# Dikkat: model dosyası büyük olabilir; çalışma dizininin doğru olduğundan emin ol.
audio_model = tf.keras.models.load_model("models/audio_model.h5")

# Modelin sınıf->index eşlemesini yükle.
# format örneği: {"normal": 0, "knock": 1, "bearing_noise": 2}
with open("models/audio_classes.json","r") as f:
    class_to_idx = json.load(f)

# index->class kolay arama için ters sözlük oluştur (modelin argmax çıktısını label'a çevirmek için)
idx_to_class = {v: k for k, v in class_to_idx.items()}


def prepare_mel_for_inference(file_stream, sr=22050, duration=4, n_mels=128):
    """
    Gelen ses verisini modele uygun mel-tensorüne dönüştürür.

    Args:
        file_stream: dosya yolu (str) veya file-like object / bytes (sf.read ile okunabilecek).
                     Eğer raw bytes göndereceksen io.BytesIO(bytes) kullan ve sf.read'e ver.
        sr: modelin beklentisi olan örnekleme hızı (sample rate). Model eğitiminde kullanılan sr ile aynı olmalı.
        duration: saniye olarak sabit örnek uzunluğu. Ses bu uzunluktan kısa ise padding, uzun ise kırpma yapılır.
        n_mels: mel bank sayısı (modelin eğitiminde kullanılanla eşleşmeli).

    Returns:
        4D numpy array: (1, frek, zaman, 1) biçiminde, modelin predict girişi için uygun.
    """

    # soundfile.read ile dosyayı oku; sf.read hem dosya yolu hem de file-like nesne alabilir.
    # Eğer file_stream bytes ise:
    #   import io
    #   file_stream = io.BytesIO(my_bytes)
    # then call sf.read(file_stream, dtype='float32')
    y, _ = sf.read(file_stream, dtype='float32')  # y -> 1D veya 2D numpy array

    # Eğer stereo veya çok kanallıysa (ndim > 1) mono'ya çevir (kanallar ortalaması)
    if y.ndim > 1:
        # axis=1 varsayılan: shape (n_samples, n_channels) ise axis 1 kanal ekseni olur
        y = np.mean(y, axis=1)

    # Modelin beklediği sabit süreyi sağlamak için padding veya kırpma uygula
    required_length = sr * duration
    if len(y) < required_length:
        # Kısa ise sıfır ile pad et (sağa doğru pad)
        y = np.pad(y, (0, required_length - len(y)))
    else:
        # Uzun ise baştan kırp: ilk required_length örnek alınır
        y = y[:required_length]

    # Mel-spektrogram hesapla (power spectrogram)
    # librosa.feature.melspectrogram döndürür: shape (n_mels, t)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)

    # Güç spektrumu dB'ye çevir (daha stabil/insan-uyumlu ölçek)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Basit min-max normalizasyonu -> 0..1 aralığı. Model eğitiminde farklı bir normalizasyon kullanıldıysa
    # burada uyumlu hale getir (ör. z-score, mean/std normalization vs).
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    # Modelin beklediği input shape'e getir:
    #  -> batch axis (1), frek/time eksenleri, kanal axis (1) => (1, n_mels, t, 1)
    return mel_db[np.newaxis, ..., np.newaxis]


def run_audio_detection(file_path):
    """
    Tek bir ses dosyası üzerinde inference yapar.

    Args:
        file_path: sf.read ile okunabilecek dosya yolu veya file-like object.

    Returns:
        dict: { "label": str, "confidence": float, "title": str, "advice": str }
    """

    # Ses verisini model girişine hazırla
    X = prepare_mel_for_inference(file_path)

    # Model tahmini: predict geri numpy array verir (batch, num_classes)
    # [0] ile batch içinden ilk örneği alıyoruz
    pred = audio_model.predict(X)[0]

    # argmax ile en yüksek olasılığa sahip sınıf index'ini bul
    idx = int(np.argmax(pred))

    # index'i gerçek label'a çevir
    label = idx_to_class[idx]

    # label için güven skoru (confidence)
    confidence = float(pred[idx])

    # Kullanıcıya döndürülecek insan-dostu açıklamalar
    # Burayı genişletebilirsin: diagnostic codes, öneriler, öncelik seviyeleri, vb.
    AUDIO_EXPLANATIONS = {
        "normal": ("Normal motor sesi", "Önemli bir anormallik tespit edilmedi."),
        "knock": ("Tıkırtı / vuruntu", "Silindir/sıkıştırma veya ateşleme kontrolü önerilir."),
        "bearing_noise": ("Rulman sesi", "Rulman kontrolü/gerekiyorsa değişimi önerilir."),
        # Modelin geri döndürebileceği diğer label'ları buraya ekle
    }

    # Eğer label mapping yoksa fallback: label metni ve genel bir uyarı
    title, advice = AUDIO_EXPLANATIONS.get(label, (label, "Öneri yok"))

    # Sonuç sözlüğü: controller veya API katmanına JSON olarak döndürülebilir
    return {
        "label": label,
        "confidence": confidence,
        "title": title,
        "advice": advice
    }