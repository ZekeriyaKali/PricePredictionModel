# app.py
# Basit bir Flask API: gelen metin, ses ve görsel verileri alır,
# ilgili inference fonksiyonlarını çağırır ve teşhis sonuçlarını JSON olarak döner.

from flask import Flask, request, jsonify  # Flask uygulaması, istek objesi ve JSON cevap oluşturma
import os                                  # Dosya yolu ve klasör oluşturma işlemleri için
import joblib                              # scikit-learn gibi modellerin diskten yüklenmesi için
from datetime import datetime              # Dosya isimlerinde zaman damgası oluşturmak için
from werkzeug.utils import secure_filename # Yüklenen dosya isimlerini güvenli hâle getirmek için (path traversal vs önlemek)

# Model inference fonksiyonlarını proje içindeki modüllerden import ediyoruz.
# Bu fonksiyonların dönüş formatları dokümante edilmiş olmalı:
#   run_image_detection(img_path) -> (detections_list, annotated_image_path)
#   run_audio_detection(audio_path) -> dict (label, confidence, title, advice)
from image_inference import run_image_detection
from audio_inference import run_audio_detection

# text model için numpy gerekli olabilir (özellikle predict'ten önce dönüştürme yapılıyorsa)
import numpy as np

# Flask uygulaması oluştur
app = Flask(__name__)

# -----------------------------------------------------
# Upload klasörleri: sunucuda gelen dosyaların saklanacağı yerler
# mkdir -p benzeri: klasör yoksa oluştur (exist_ok=True -> hata verme)
# -----------------------------------------------------
UPLOAD_DIRS = ["uploads/audio", "uploads/images", "uploads/annotated"]
for d in UPLOAD_DIRS:
    os.makedirs(d, exist_ok=True)  # Klasör zaten varsa üzerine yazma/ata hata verme

# -----------------------------------------------------
# Text (metin) modeli ve vectorizer (TF-IDF vb.) yükleme
# Hata durumunda değişkenleri None yapıp uygulamanın ayakta kalmasını sağlıyoruz
# (geliştirme: hata log'la, prod: monitoring/alert ekle)
# -----------------------------------------------------
try:
    text_model = joblib.load("models/fault_model.pkl")      # scikit-learn tipi model
    vectorizer = joblib.load("models/vectorizer.pkl")      # text -> feature dönüşümü (örn. TfidfVectorizer)
except Exception as e:
    # Model yüklenemezse uyarı yaz ve değişkenleri None olarak ayarla
    print(f"⚠️ Text model yüklenemedi: {e}")
    text_model, vectorizer = None, None

# -----------------------------------------------------
# /diagnose endpoint: multipart/form-data ile description, audio, image alır
# - description: form field (string)
# - audio: file field (wav/mp3)
# - image: file field (jpg/png)
# -----------------------------------------------------
@app.route("/diagnose", methods=["POST"])
def diagnose():
    # Form verilerini al
    description = request.form.get("description")      # textarea veya input ile gelen metin
    audio = request.files.get("audio")                 # audio dosyası (werkzeug FileStorage) veya None
    image = request.files.get("image")                 # image dosyası veya None

    # Döndürülecek JSON şablonu: her alan başlangıçta None veya empty list
    response = {
        "text": None,
        "audio": None,
        "image": None,
        "annotated_image": None,
        "final_recommendations": []
    }

    try:
        # -----------------------------
        # 1) TEXT ANALYSIS (opsiyonel)
        # -----------------------------
        # Eğer açıklama gönderilmiş ve text model + vectorizer başarılı yüklenmişse işle
        if description and text_model and vectorizer:
            # Metni vectorizer ile feature vektörüne çevir
            X = vectorizer.transform([description])
            # Model ile tahmin al
            pred = text_model.predict(X)[0]
            # JSON'da dönmek üzere label ve genel advice ekle
            response["text"] = {
                "label": pred,
                "advice": "Metin tabanlı öneri (ör: servise götürün, şu parçayı kontrol edin...)"
            }

        # -----------------------------
        # 2) AUDIO ANALYSIS (opsiyonel)
        # -----------------------------
        # Eğer kullanıcı bir audio dosyası yüklediyse
        if audio:
            # Güvenli bir dosya adı oluştur: timestamp + secure_filename
            audio_filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(audio.filename)
            audio_path = os.path.join("uploads/audio", audio_filename)

            # Dosyayı diske kaydet
            audio.save(audio_path)

            # run_audio_detection fonksiyonunu çağır; bu fonksiyon model inference yapar
            # Beklenen dönüş ör: {"label": "knock", "confidence":0.92, "title": "...", "advice":"..."}
            audio_result = run_audio_detection(audio_path)

            # API cevabına hem dosya yolu hem de inference sonucunu ekle
            response["audio"] = {"file": audio_path, **audio_result}

        # -----------------------------
        # 3) IMAGE ANALYSIS (opsiyonel)
        # -----------------------------
        # Eğer kullanıcı bir görsel yüklediyse
        if image:
            img_filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(image.filename)
            img_path = os.path.join("uploads/images", img_filename)

            # Görseli diske kaydet
            image.save(img_path)

            # run_image_detection fonksiyonunu çağır; genelde (detections, annotated_path) döndürür
            # detections: [{ "label": "...", "title":"...", "advice":"..." }, ...]
            # annotated_path: sunucuda işaretlenmiş/çizilmiş görselin yolu (örn. uploads/annotated/...)
            detections, annotated_path = run_image_detection(img_path)

            # JSON cevabına ekle
            response["image"] = {"file": img_path, "detections": detections}
            response["annotated_image"] = annotated_path

        # -----------------------------
        # 4) FINAL RECOMMENDATIONS (özet öneriler)
        # -----------------------------
        # Görsel tespitlerinden genel öneri metinleri oluştur
        if response["image"] and response["image"].get("detections"):
            for d in response["image"]["detections"]:
                # her detection için title ve advice’i final önerilere ekle
                # (ör: "Kırık hortum: Sızıntı kontrolü önerilir")
                response["final_recommendations"].append(f"{d['title']}: {d['advice']}")

        # Ses analizinden gelen öneri varsa ekle
        if response["audio"]:
            response["final_recommendations"].append(
                f"{response['audio'].get('title', 'Audio')} : {response['audio'].get('advice', '')}"
            )

        # Metin analiz sonucu varsa ekle
        if response["text"]:
            response["final_recommendations"].append(f"Text result: {response['text']['label']}")

    except Exception as e:
        # Herhangi bir hata olursa (dosya IO, model hatası vs.) 500 döndür ve hata mesajını JSON içinde ver
        # Prod ortamında hata detaylarını logla fakat kullanıcıya generic hata göster.
        return jsonify({"error": str(e)}), 500

    # Başarılıysa toplanmış response'u JSON olarak dön
    return jsonify(response)


# Debug/Development sunucusu
if __name__ == "__main__":
    # app.run parametrelerini production için gunicorn/uwsgi gibi WSGI sunucularla değiştir
    app.run(port=5001, debug=True)
