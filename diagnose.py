from flask import Flask, request, jsonify
import joblib, os, json
from datetime import datetime
from image_inference import run_image_detection
from audio_inference import run_audio_detection
# text model
import pickle
import numpy as np

app = Flask(__name__)
os.makedirs("uploads/audio", exist_ok=True)
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/annotated", exist_ok=True)

# load text model
text_model = joblib.load("models/fault_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    description = request.form.get("description")
    audio = request.files.get("audio")
    image = request.files.get("image")

    response = {
        "text": None,
        "audio": None,
        "image": None,
        "annotated_image": None
    }

    # TEXT
    if description:
        X = vectorizer.transform([description])
        pred = text_model.predict(X)[0]
        response["text"] = {"label": pred, "advice": "Metin tabanlı öneri (detaylı harita yazılabilir)"}

    # AUDIO
    if audio:
        # save audio
        audio_filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + audio.filename
        audio_path = os.path.join("uploads/audio", audio_filename)
        audio.save(audio_path)
        audio_result = run_audio_detection(audio_path)
        response["audio"] = {"file": audio_path, **audio_result}

    # IMAGE
    if image:
        img_filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + image.filename
        img_path = os.path.join("uploads/images", img_filename)
        image.save(img_path)
        detections, annotated_path = run_image_detection(img_path)
        response["image"] = {"file": img_path, "detections": detections}
        response["annotated_image"] = annotated_path

    # combine logic (basit - örnek)
    # ör: eğer image detection "hose_tear" varsa öncelik ver
    final_recommendations = []
    if response["image"] and response["image"]["detections"]:
        for d in response["image"]["detections"]:
            final_recommendations.append(f"{d['title']}: {d['advice']}")
    if response["audio"]:
        final_recommendations.append(f"{response['audio']['title']}: {response['audio']['advice']}")
    if response["text"]:
        final_recommendations.append(f"Text result: {response['text']['label']}")

    response["final_recommendations"] = final_recommendations

    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5001, debug=True)