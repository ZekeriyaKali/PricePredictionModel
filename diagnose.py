from flask import Flask, request, jsonify
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Upload klasörleri
os.makedirs("uploads/audio", exist_ok=True)
os.makedirs("uploads/images", exist_ok=True)


# Modeli yükle
model = joblib.load("fault_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    description = request.form.get("description")
    audio = request.files.get("audio")
    image = request.files.get("image")

    results = {}

    # ✅ Text analizi
    if description:
        X = vectorizer.transform([description])
        prediction = model.predict(X)[0]
        results["diagnosis"] = prediction
    else:
        results["diagnosis"] = "Metin girişi yok"

    # ✅ Ses dosyasını kaydet
    if audio:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + audio.filename
        filepath = os.path.join("uploads/audio", filename)
        audio.save(filepath)
        results["audio_file"] = filepath
    else:
        results["audio_file"] = None

    # ✅ Resim dosyasını kaydet
    if image:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + image.filename
        filepath = os.path.join("uploads/images", filename)
        image.save(filepath)
        results["image_file"] = filepath
    else:
        results["image_file"] = None

    return jsonify(results)


if __name__ == "__main__":
    app.run(port=5001, debug=True)