from flask import Flask, request, jsonify
import os
import joblib
from datetime import datetime
from werkzeug.utils import secure_filename

# Model importları
from image_inference import run_image_detection
from audio_inference import run_audio_detection

# text model
import numpy as np

app = Flask(__name__)

# Klasörlerin oluşturulması
UPLOAD_DIRS = ["uploads/audio", "uploads/images", "uploads/annotated"]
for d in UPLOAD_DIRS:
    os.makedirs(d, exist_ok=True)

# Text model ve vectorizer yükle
try:
    text_model = joblib.load("models/fault_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
except Exception as e:
    print(f"⚠️ Text model yüklenemedi: {e}")
    text_model, vectorizer = None, None


@app.route("/diagnose", methods=["POST"])
def diagnose():
    description = request.form.get("description")
    audio = request.files.get("audio")
    image = request.files.get("image")

    response = {
        "text": None,
        "audio": None,
        "image": None,
        "annotated_image": None,
        "final_recommendations": []
    }

    try:
        # TEXT ANALYSIS
        if description and text_model and vectorizer:
            X = vectorizer.transform([description])
            pred = text_model.predict(X)[0]
            response["text"] = {
                "label": pred,
                "advice": "Metin tabanlı öneri (ör: servise götürün, şu parçayı kontrol edin...)"
            }

        # AUDIO ANALYSIS
        if audio:
            audio_filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(audio.filename)
            audio_path = os.path.join("uploads/audio", audio_filename)
            audio.save(audio_path)

            audio_result = run_audio_detection(audio_path)
            response["audio"] = {"file": audio_path, **audio_result}

        # IMAGE ANALYSIS
        if image:
            img_filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(image.filename)
            img_path = os.path.join("uploads/images", img_filename)
            image.save(img_path)

            detections, annotated_path = run_image_detection(img_path)
            response["image"] = {"file": img_path, "detections": detections}
            response["annotated_image"] = annotated_path

        # FINAL RECOMMENDATIONS
        if response["image"] and response["image"].get("detections"):
            for d in response["image"]["detections"]:
                response["final_recommendations"].append(f"{d['title']}: {d['advice']}")

        if response["audio"]:
            response["final_recommendations"].append(
                f"{response['audio'].get('title', 'Audio')} : {response['audio'].get('advice', '')}"
            )

        if response["text"]:
            response["final_recommendations"].append(f"Text result: {response['text']['label']}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(response)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
