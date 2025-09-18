# train_car_price_model.py
# Bu script: car_data.csv dosyasından veriyi okuyup ön işlem uygulayarak
# RandomForestRegressor içeren bir Pipeline oluşturur, modeli eğitir, test eder ve kaydeder.

import pandas as pd                      # Veri okuma / DataFrame işlemleri için
from sklearn.model_selection import train_test_split  # Eğitim/test bölme
from sklearn.ensemble import RandomForestRegressor    # Regresyon için Random Forest
from sklearn.preprocessing import OneHotEncoder       # Kategorik değişkenlerin one-hot dönüşümü
from sklearn.compose import ColumnTransformer         # Kolon bazlı dönüşümler (sayısal / kategorik ayırma)
from sklearn.pipeline import Pipeline                 # Preprocessing + model pipeline'ı
from sklearn.metrics import r2_score, mean_absolute_error  # Performans ölçümleri
import joblib                                # Modeli disk'e kaydetmek / yüklemek için
import os                                     # Dosya/dizin path işlemleri için

# ----------------------------------------
# Dosya/dizin referansını güvenli şekilde al
# ----------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir -> script'in çalıştığı dizin. Mutlak path alarak dosya referanslarını güvenli yapıyoruz.

# ----------------------------------------
# 1) Veriyi oku
# - car_data.csv dosyasını okunur. Sep olarak ';' kullanılmış.
# - os.path.join ile mutlak/dinamik path oluşturuyoruz (platform bağımsız).
# ----------------------------------------
df = pd.read_csv(os.path.join(current_dir, "car_data.csv"), sep=";")

# ----------------------------------------
# 2) Veri tiplerini doğrula / dönüştür
# - CSV'den gelen Year/Km/Price sütunları string olabilir => sayısala çevir.
# - errors="coerce" ile dönüşmezse NaN olur; sonrasında NaN için imputation gerekebilir.
# ----------------------------------------
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")   # Yıl sütununu numeric yap
df["Km"] = pd.to_numeric(df["Km"], errors="coerce")       # Kilometre sütununu numeric yap
df["Price"] = pd.to_numeric(df["Price"], errors="coerce") # Fiyat sütununu numeric yap

# (OPSİYONEL) Burada NaN değer kontrolü / temizleme yapılması önerilir:
#   df.dropna(subset=["Year","Km","Price"], inplace=True)  veya uygun imputation uygulayın.

# ----------------------------------------
# 3) Giriş (X) ve hedef (y) sütunlarını ayır
# ----------------------------------------
X = df.drop("Price", axis=1)  # Tahmin edilecek değişken hariç tüm kolonlar girdi olarak kullanılır
y = df["Price"]               # Hedef değişken: Price

# ----------------------------------------
# 4) Kategorik sütunları tanımla
# - Bu sütunlar OneHotEncoder ile dönüştürülecek.
# - Eğer ek kategorik kolonlar varsa buraya ekle.
# ----------------------------------------
categorical_features = ["Brand", "Model", "GearType", "FuelType", "City"]

# ----------------------------------------
# 5) Preprocessor: ColumnTransformer ile pipeline hazırlama
# - "cat" transformer: OneHotEncoder(handle_unknown="ignore")
#      -> eğitim sırasında görülmeyen kategorilere karşı dayanıklı.
# - remainder='passthrough' -> sayısal sütunlar (Year, Km vb.) bırakılır.
# ----------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder='passthrough'  # Diğer (sayısal) kolonlar olduğu gibi pipeline'a geçsin
)

# ----------------------------------------
# 6) Pipeline: preprocessor + model
# - Pipeline kullanmak, preprocessing ve modeli tek bir obje gibi tutar.
# - Böylece .fit/.predict kolay olur ve üretime alırken tek obje kaydedilir.
# ----------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,   # ağaç sayısı; artırmak stabiliteyi artırır ama hesap maliyetini yükseltir
        random_state=42
    ))
])

# ----------------------------------------
# 7) Eğitim / Test bölme
# - test_size=0.2 => verinin %20'si test için ayrılır
# - random_state reproducibility içindir
# - NOT: stratify burada kullanılmadı; kategorik dağılım dengesizse stratify=y önerilir.
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# 8) Modeli eğit
# - Dikkat: orijinal kodda model.fit(X, y) olarak tüm veriyle eğitim yapılmış.
#   Bu satır kodu değiştirmeden bıraktım, ancak genelde doğru kullanım:
#       model.fit(X_train, y_train)
#   olmalıdır (sadece eğitim setinde fit etmek).
# - Aşağıdaki satır orijinal haliyle bırakıldı; istersen X_train,y_train olarak değiştir.
# ----------------------------------------
model.fit(X, y)  # <-- öneri: model.fit(X_train, y_train)

# ----------------------------------------
# 9) Tahmin yap ve performans ölç
# - Test seti üzerinde tahmin alıp R^2 ve MAE ile değerlendirme yapıyoruz.
# - R^2: açıklanan varyans; MAE: ortalama mutlak hata (gerçek para birimi açısından yorumlanabilir).
# ----------------------------------------
y_pred = model.predict(X_test)
print("R² Skoru:", round(r2_score(y_test, y_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# ----------------------------------------
# 10) Modeli kaydet
# - joblib.dump ile pipeline (preprocessor + regressor) tek dosyada saklanır.
# - production'da modeli yüklemek için joblib.load kullan.
# ----------------------------------------
model_path = os.path.join(current_dir, "car_price_model.pkl")
joblib.dump(model, model_path)

print("Model başarıyla kaydedildi:", model_path)