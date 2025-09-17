# app.py
# Bu Flask API, eğitilmiş makine öğrenmesi modelini kullanarak araç fiyat tahmini yapar.
# Aynı zamanda tahmin sonuçlarını SQL Server veritabanına kaydeder.

from datetime import datetime
from flask import Flask, request, jsonify
import joblib                   # Model dosyasını yüklemek için
import pandas as pd             # Veri işleme
import pyodbc                   # SQL Server bağlantısı için
import urllib                   # Connection string encode etmek için
from sqlalchemy import create_engine, text   # SQLAlchemy ile veritabanı işlemleri

# -----------------------------------------------------
# Flask uygulamasını başlat
# -----------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------
# PyODBC ile temel bağlantı testi
# -----------------------------------------------------
conn = pyodbc.connect(
    r'Driver={ODBC Driver 17 for SQL Server};Server=(localdb)\MSSQLLocalDB;Database=DriveList;Trusted_Connection=yes;'
)
print("Bağlantı başarılı!")

# -----------------------------------------------------
# SQLAlchemy engine ayarı
# Connection string encode edilerek güvenli hale getirilir
# -----------------------------------------------------
connection_string = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=(localdb)\\MSSQLLocalDB;"
    "Database=DriveList;"
    "Trusted_Connection=yes;"
)
params = urllib.parse.quote_plus(connection_string)  # URL encode

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# -----------------------------------------------------
# SQLAlchemy connection test
# -----------------------------------------------------
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Bağlantı başarılı:", result.scalar())
except Exception as e:
    print("❌ Bağlantı hatası:", e)

# -----------------------------------------------------
# Eğitilmiş model dosyasını yükle
# -----------------------------------------------------
model = joblib.load('car_price_model.pkl')  # Pickle edilmiş ML modeli yükler
print("Model dosyası başarıyla yüklendi!")

# -----------------------------------------------------
# /predict endpoint
# -----------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # 1) Gelen JSON verisini al
    data = request.get_json()
    # 2) DataFrame’e çevir (tek kayıt bile olsa DataFrame yapılır)
    df = pd.DataFrame([data])

    # 3) Flask’tan gelen kolon isimlerini modele uygun olacak şekilde rename et
    df.rename(columns={
        'brand': 'Brand',
        'model': 'Model',
        'year': 'Year',
        'km': 'Km',
        'gearType': 'GearType',
        'fuelType': 'FuelType',
        'city': 'City'
    }, inplace=True)

    # Debug amaçlı gelen veriyi logla
    print("🧾 Gelen veri:", df.to_dict(orient='records'))

    # 4) Model tahminini yap
    prediction = model.predict(df)[0]
    print("📊 Yapılan tahmin:", prediction)

    # 5) Tahmini SQL Server’a kaydet
    with engine.begin() as conn:
        query = text("""
            INSERT INTO Predictions (Brand, Model, Year, Km, GearType, FuelType, City, PredictedPrice, CreatedAt)
            VALUES (:Brand, :Model, :Year, :Km, :GearType, :FuelType, :City, :PredictedPrice, :CreatedAt)
        """)
        conn.execute(query, {
            "Brand": df["Brand"][0],
            "Model": df["Model"][0],
            "Year": int(df["Year"][0]),
            "Km": int(df["Km"][0]),
            "GearType": df["GearType"][0],
            "FuelType": df["FuelType"][0],
            "City": df["City"][0],
            "PredictedPrice": float(prediction),
            "CreatedAt": datetime.now()
        })

    # 6) Tahmini JSON response olarak döndür
    return jsonify({'PricePrediction': round(prediction, 2)})

# -----------------------------------------------------
# Flask uygulamasını ayağa kaldır
# -----------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)   # Debug açık, API 5000 portunda çalışır
