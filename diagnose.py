from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/diagnose", methods=["POST"])
def diagnose():
    description = request.form.get("description")  # metin
    audio = request.files.get("audio")            # ses dosyası
    image = request.files.get("image")            # resim

    # Buraya AI model entegrasyonu eklenecek
    result = "Arıza analizi yapıldı. Test cevabı: "
    if description:
        result += f"Metin: {description}. "
    if audio:
        result += f"Ses dosyası alındı: {audio.filename}. "
    if image:
        result += f"Görüntü dosyası alındı: {image.filename}. "

    return jsonify({"diagnosis": result})

if __name__ == "__main__":
    # Servisi http://localhost:5001 üzerinde çalıştırır
    app.run(host="0.0.0.0", port=5001, debug=True)