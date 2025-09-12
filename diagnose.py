from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Modeli yükle
model = joblib.load("fault_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    description = request.form.get("description")
    audio = request.files.get("audio")
    image = request.files.get("image")

    if not description and not audio and not image:
        return jsonify({"error": "Lütfen en az bir veri girin"}), 400

    result_parts = []

    # Text analizi
    if description:
        X = vectorizer.transform([description])
        prediction = model.predict(X)[0]
        result_parts.append(f"Metin analizi sonucu: {prediction}")

    # Ses dosyası işleme (örnek placeholder)
    if audio:
        result_parts.append(f"Ses dosyası alındı: {audio.filename}")

    # Resim dosyası işleme (örnek placeholder)
    if image:
        result_parts.append(f"Resim alındı: {image.filename}")

    return jsonify({"diagnosis": " | ".join(result_parts)})

if __name__ == "__main__":
    app.run(port=5001, debug=True)