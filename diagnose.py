from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Modeli yükle
model = joblib.load("fault_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    description = request.form.get("description")

    if not description:
        return jsonify({"error": "Lütfen bir açıklama girin"}), 400

    X = vectorizer.transform([description])
    prediction = model.predict(X)[0]

    return jsonify({"diagnosis": prediction})

if __name__ == "__main__":
    app.run(port=5001, debug=True)