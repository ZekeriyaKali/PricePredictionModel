from datetime import datetime

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pyodbc
import urllib # Get encode from URL
from sqlalchemy import create_engine, text

app = Flask(__name__)

conn = pyodbc.connect(
    r'Driver={ODBC Driver 17 for SQL Server};Server=(localdb)\MSSQLLocalDB;Database=DriveList;Trusted_Connection=yes;'
)
print("Bağlantı başarılı!")

connection_string = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=(localdb)\\MSSQLLocalDB;"
    "Database=DriveList;"
    "Trusted_Connection=yes;"
)
params = urllib.parse.quote_plus(connection_string)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Bağlantı başarılı:", result.scalar())  #sqlalchemy connection test
except Exception as e:
    print("❌ Bağlantı hatası:", e)

# Eğittiğin model dosyasını yükle
model = joblib.load('car_price_model.pkl')  # Örnek model dosyan
print("Model dosyası başarıyla yüklendi!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])  # Tek bir veri için
    df.rename(columns={
        'brand': 'Brand',
        'model': 'Model',
        'year': 'Year',
        'km': 'Km',
        'gearType': 'GearType',
        'fuelType': 'FuelType',
        'city': 'City'
    }, inplace=True)

    print("🧾 Gelen veri:", df.to_dict(orient='records'))  # 👈 bunu ekle


    prediction = model.predict(df)[0]
    print("📊 Yapılan tahmin:", prediction)

    # 📌 Veritabanına kaydet
    with engine.begin() as conn:
        query = text("""
               INSERT INTO Predictions (Brand, Model, Year, Km, GearType, FuelType, City, PredictedPrice,CreatedAt)
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

    return jsonify({'PricePrediction': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
