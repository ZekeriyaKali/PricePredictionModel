import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Veriyi oku
df = pd.read_csv(os.path.join(current_dir, "car_data.csv"), sep=";")

# 2. Veri tiplerini kontrol et
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Km"] = pd.to_numeric(df["Km"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Giriş ve çıkış sütunları
X = df.drop("Price", axis=1)
y = df["Price"]

# Kategorik sütunlar
categorical_features = ["Brand", "Model", "GearType", "FuelType", "City"]

# Sayısal sütunlar otomatik olarak kullanılacak

# Pipeline oluştur
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder='passthrough'  # Sayısal veriler olduğu gibi geçsin
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modeli eğit
model.fit(X, y)

# 9. Tahmin yap ve performans ölç
y_pred = model.predict(X_test)
print("R² Skoru:", round(r2_score(y_test, y_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# Modeli kaydet
model_path = os.path.join(current_dir, "car_price_model.pkl")
joblib.dump(model, model_path)

print("Model başarıyla kaydedildi.")