import os                                     # Dosya/dizin işlemleri (klasör var mı kontrol, oluşturma vb.) için kullanılır.
import librosa                                 # Ses dosyalarını okumak ve mel-spektrogram gibi özellikleri çıkarmak için popüler kütüphane.
import numpy as np                             # NumPy: sayısal hesaplama, dizi manipülasyonu, padding vb. işlemler için.
from tensorflow.keras import layers, models    # Keras: model tanımlama (katmanlar) ve model objesi (Sequential vb.) için.
from sklearn.model_selection import train_test_split  # Eğitim/validasyon bölme işlemi için.
from tensorflow.keras.utils import to_categorical      # Etiketleri one-hot vektörlerine çevirmek için.
from glob import glob                           # Dosya desenleriyle dosya listesini almak için (örn. klasördeki tüm .wav dosyaları).
import json                                     # Sınıf-idx mapping'ini kaydetmek/okumak için JSON kullanıyoruz.

# -------------------------
# Mel spectrogram çıkarma fonksiyonu
# -------------------------
def extract_mel(path, sr=22050, n_mels=128, duration=4):
    """
    Verilen WAV dosyasından mel-spektrogram çıkarır ve normalize eder.
    - path: ses dosyasının yolu (string)
    - sr: sample rate; model eğitilirken kullanılan ile aynı olmalı (genelde 22050)
    - n_mels: kaç mel bandı kullanılacak
    - duration: sabit süre (saniye). Kısa dosyalar pad edilir, uzunlar kırpılır.
    Dönen değer: (n_mels, time_steps) şeklinde bir numpy array (float32).
    """

    y, _ = librosa.load(path, sr=sr, duration=duration)
    # librosa.load: dosyayı oku, dönüş sr parametresine göre yeniden örnekle (resample).
    # duration parametresi: sadece başlangıç kısmını alır; uzun dosyalar için kısaltır.

    if len(y) < sr * duration:
        # Eğer okunan ses örnek sayısı modelin beklediğinden kısa ise
        # sonuna sıfır ile pad ederek sabit uzunluk sağlanır.
        y = np.pad(y, (0, sr * duration - len(y)))
        # pad: (before, after) -> burada başa 0, sona eksik kadar pad.

    # Mel-spektrogram: güç spektrumu tabanlı mel katsayıları
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    # Çıktı shape'i: (n_mels, time_steps)

    # dB'ye dönüştürme: power_to_db, log-ölçeğe çevirir -> model için daha stabil
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Min-max normalizasyon: 0..1 aralığına sıkıştırma
    # NOT: Eğer eğitim tarafında farklı normalizasyon (ör: mean/std) kullanıldıysa,
    # burada da aynı yöntem uygulanmalı.
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    return mel_db.astype(np.float32)  # float32 ile tip uyumu sağlanır (TensorFlow için uygun)

# -------------------------
# Veri yükleme ve etiketleme
# -------------------------
X = []   # Mel spectrogram dizilerini tutacak liste (her eleman: 2D array)
y = []   # Etiketlerin integer karşılıklarını tutacak liste

# Veri klasöründeki sınıf isimlerini (alt klasörlar) oku ve alfabetik sırala
# Örnek beklenen klasör yapısı: audio_dataset/normal/*.wav, audio_dataset/knock/*.wav, ...
classes = sorted(os.listdir("audio_dataset"))
# classes artık ['bearing_noise', 'knock', 'normal', ...] gibi bir liste döner

# class -> index mapping oluştur (model eğitimi ve inference sırasında aynı mapping kullanılmalı)
class_to_idx = {c: i for i, c in enumerate(classes)}
# Ör: {'bearing_noise': 0, 'knock': 1, 'normal': 2}

# Her sınıf için ilgili klasördeki .wav dosyalarını gez
for cls in classes:
    # glob ile tüm wav dosyalarını al; alt klasör yolunu oluştur
    pattern = os.path.join("audio_dataset", cls, "*.wav")
    for fpath in glob(pattern):
        # Dosya başına mel çıkar
        mel = extract_mel(fpath)
        # Listeye ekle
        X.append(mel)
        # Etiket olarak sınıf index'ini ekle
        y.append(class_to_idx[cls])
        # Not: büyük datasetlerde burayı progress bar (tqdm) ile sarmak faydalıdır

# Numpy dizisine çevir ve model için kanal dimension ekle
# X: shape (n_samples, n_mels, time_steps) -> -> [..., np.newaxis] ile (n, n_mels, time, 1)
X = np.array(X)[..., np.newaxis]

# Etiketleri one-hot encode et (categorical_crossentropy için)
y = to_categorical(y, num_classes=len(classes))

# -------------------------
# Eğitim / doğrulama bölünmesi
# -------------------------
# Reproducibility için random_state verildi (aynı bölünme tekrar üretilebilir)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)
# Eğer sınıf dengesizliği varsa stratify parametresi ile y (sınıflar) üzerinden stratify etmek iyi olur.

# -------------------------
# Model mimarisi (basit CNN)
# -------------------------
model = models.Sequential([
    # Giriş katmanı: X_train.shape[1:] -> (n_mels, time_steps, 1)
    layers.Input(shape=X_train.shape[1:]),

    # 1. Convolution bloğu: 16 filtre, 3x3 kernel, ReLU aktivasyon, 'same' ile kenar koruması
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    # Downsampling: boyutları yarıya indir (zaman veya frekans ekseninde)
    layers.MaxPool2D((2, 2)),

    # 2. Convolution bloğu: daha fazla filtre (32)
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),

    # 3. Convolution bloğu: daha yüksek seviye feature'lar (64 filtre)
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    # GlobalAveragePooling tüm feature haritalarını ortalayarak sabit boyutlu vektör elde eder
    # Bu, parametre sayısını azaltır ve overfitting riskini düşürür
    layers.GlobalAveragePooling2D(),

    # Dense katmanı: öğrenilen özelliklerden sınıflandırma öncesi temsil çıkar
    layers.Dense(64, activation='relu'),

    # Dropout: overfitting'i engellemek için rastgele nöron kapatma
    layers.Dropout(0.3),

    # Çıkış katmanı: sınıf sayısı kadar nöron, softmax ile olasılık üretir
    layers.Dense(len(classes), activation='softmax')
])

# -------------------------
# Model derleme
# -------------------------
# Optimizer: Adam (varsayılan öğrenme hızı iyi başlangıç), loss: categorical_crossentropy
# Metric olarak accuracy takip edilecek
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# Model eğitimi
# -------------------------
# epochs: kaç kez tüm veri üzerinden geçilecek; batch_size: her adımda işlenecek örnek sayısı
# validation_data ile eğitim sırasında val performansı izlenir
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16
)
# 'history' objesi ile eğitim/validasyon loss ve accuracy'lerini saklayıp sonrasında görselleyebilirsin.

# -------------------------
# Eğitilmiş modelin kaydedilmesi
# -------------------------
os.makedirs("models", exist_ok=True)         # models klasörü yoksa oluştur
model.save("models/audio_model.h5")          # Keras .h5 formatında modeli kaydet

# Sınıf mapping'ini JSON olarak kaydet (inference tarafında idx->label dönmek için)
with open("models/audio_classes.json", "w") as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

# Eğitim tamamlandığını bildir
print("✅ Model ve class mapping başarıyla kaydedildi!")
