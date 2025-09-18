# train_audio_model.py
# Bu script: audio_dataset klasöründeki .wav dosyalarından MFCC özellikleri çıkarır,
# bir RandomForest sınıflandırıcı eğitir, değerlendirme raporu yazdırır ve modeli disk'e kaydeder.

import os                             # Dosya/dizin işlemleri (listeleme, path birleştirme, vs.)
import numpy as np                    # Sayısal hesaplamalar, dizi manipülasyonları için
import librosa                        # Ses işleme: yükleme, MFCC gibi feature çıkarımı için
import joblib                         # scikit-learn modellerini serileştirmek/kaydetmek için
from sklearn.model_selection import train_test_split  # Veri bölme (train/test)
from sklearn.ensemble import RandomForestClassifier    # Model sınıfı (rastgele orman)
from sklearn.metrics import classification_report      # Performans raporu

# -------------------------
# Sabitler / dataset yolu
# -------------------------
# Beklenen klasör yapısı örneği:
# audio_dataset/
#   normal/
#   kayis_gicirti/
#   motor_vuruntu/
DATASET_PATH = "audio_dataset"   # Veri setinin bulunduğu ana dizin (değiştirilebilir)

# X: özellik matrisini tutacak liste, y: etiket (label) listesini tutacak
X, y = [], []

# -------------------------
# Öznitelik çıkarma fonksiyonu
# -------------------------
def extract_features(file_path):
    """
    Bir .wav dosyasından MFCC temelli özellik çıkarır.
    - file_path: wav dosyasının yolu
    Döndürür: tek boyutlu ortalama MFCC vektörü (örneğin 40 özellik)
    Not: Model veya pipeline farklı feature bekliyorsa (mel, chroma, zero-crossing vb.)
    burayı uyarlamak gerekir.
    """
    # librosa.load: dosyayı oku. sr=None ile orijinal sample rate korunur (yeniden örnekleme yapılmaz).
    # Eğer tüm dosyaları aynı sr'e getirmek istersen sr=22050 gibi sabit bir değer ver.
    y, sr = librosa.load(file_path, sr=None)

    # MFCC çıkar: n_mfcc parametresi kaç MFCC alınacağını belirler.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Zaman boyutu boyunca ortalama al (her dosya için sabit uzunluklu feature vektörü üretmek için).
    # Alternatif: std, max, min veya frame-level pooling de kullanılabilir.
    return np.mean(mfcc.T, axis=0)

# -------------------------
# Veri yükleme: klasörleri dolaş ve .wav dosyalarını işle
# -------------------------
# os.listdir(DATASET_PATH) ile sınıf isimlerini (alt klasörları) alıyoruz.
for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)  # dosya yolunu güvenli şekilde birleştir
    if not os.path.isdir(folder):
        # Eğer listedeki bir öğe dosya ise (klasör değilse) atla
        continue

    # Her klasördeki dosyaları dön
    for file in os.listdir(folder):
        # Sadece .wav uzantılı dosyaları işle (büyük veri setlerinde filtreleme önemli)
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)     # tam dosya yolu
            features = extract_features(file_path)     # MFCC vb. özellik çıkar
            X.append(features)                         # özellikleri listeye ekle
            y.append(label)                            # etiketi ekle (klasör ismi label olarak kullanılıyor)

# Listeleri NumPy dizilerine çevir (scikit-learn ile uyumlu format)
X, y = np.array(X), np.array(y)

# -------------------------
# Eğitim / Test bölme
# -------------------------
# test_size=0.2 => verinin %20'si test için ayrılır
# random_state sabitlenirse aynı bölünme tekrar üretilebilir (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Model oluşturma ve eğitme
# -------------------------
# RandomForestClassifier: güçlü, ayarları nispeten basit, iyi başlangıç modeli
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit ile eğitim verisi üzerinde öğrenme yapılır
model.fit(X_train, y_train)

# -------------------------
# Test / Değerlendirme
# -------------------------
# Eğitilmiş model ile test seti üzerindeki tahminleri al
y_pred = model.predict(X_test)

# classification_report: precision, recall, f1-score ve support (sınıf başına örnek sayısı) verir
# Konsola yazdırmak, hızlı kalite kontrolü için faydalıdır
print(classification_report(y_test, y_pred))

# -------------------------
# Modeli disk'e kaydetme
# -------------------------
# joblib.dump model nesnesini dosyaya seri hale getirir (pickle benzeri)
# Kaydedilen dosyayı production'da load ederek predict() çağrılabilir
joblib.dump(model, "audio_model.pkl")
print("✅ Audio model kaydedildi: audio_model.pkl")