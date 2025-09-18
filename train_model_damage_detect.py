# train_text_model.py
# Bu script: 'car_faults.csv' içindeki 'description' -> 'issue' verisini kullanarak
# TF-IDF + Logistic Regression tabanlı bir metin sınıflandırıcı eğitir ve model + vectorizer'ı kaydeder.

# pandas: CSV okuma, DataFrame işlemleri (veri hazırlama için)
import pandas as pd

# train_test_split: veriyi eğitim ve test olarak ayırmak için
from sklearn.model_selection import train_test_split

# TF-IDF vektörizer: metinleri sayısal özelliklere çevirir (n-gram desteği ile)
from sklearn.feature_extraction.text import TfidfVectorizer

# LogisticRegression: sınıflandırma algoritması (lineer, hızlı ve iyi baseline)
from sklearn.linear_model import LogisticRegression

# Performans ölçümleri: classification_report (precision, recall, f1) ve accuracy
from sklearn.metrics import classification_report, accuracy_score

# joblib: sklearn objelerini (model, vectorizer) diske kaydetmek için kullanılır
import joblib

# ---------------------------------------------------------
# 1) Veri yükleme
# - CSV dosyasını ';' ile ayrılmış olarak okuyoruz (senin dosyanda bu delimiter kullanılmış).
# - İlk birkaç satırı ve kolon isimlerini yazdırarak veri yapısını doğruluyoruz.
# ---------------------------------------------------------
df = pd.read_csv("car_faults.csv", delimiter=";")  # CSV dosyasını DataFrame'e yükle
print("Veri örneği:")                              # debug/inceleme amaçlı
print(df.head())                                   # veri örneklerini göster
print("Kolonlar:", df.columns)                     # kolon isimlerini yazdır

# ---------------------------------------------------------
# 2) Train/Test ayrımı
# - description (metin) bağımsız değişken (X), issue (etiket) bağımlı değişken (y)
# - test_size=0.2 => verinin %20'si test için ayrılır
# - random_state sabitlenirse sonuçlar tekrarlanabilir (reproducibility)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["description"],    # özellik kolonu (text)
    df["issue"],          # hedef kolonu (etiket)
    test_size=0.2,        # test oranı
    random_state=42       # tekrar üretilebilirlik için sabit tohum
)

# ---------------------------------------------------------
# 3) Metni sayısallaştırma (TF-IDF)
# - ngram_range=(1,2) => unigram + bigram al; kelime kombinasyonları modelin daha iyi öğrenmesini sağlar
# - fit_transform: eğitim verisinde fit ederek vektörize et (vocab oluşturur)
# - transform: test verisini aynı vocabulary ile dönüştür
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2))   # TF-IDF vektörleştirici örneği
X_train_tfidf = vectorizer.fit_transform(X_train)  # sadece eğitim verisinde fit et ve dönüştür
X_test_tfidf = vectorizer.transform(X_test)        # test verisini aynı vektörizer ile dönüştür

# ---------------------------------------------------------
# 4) Model eğitimi (Logistic Regression)
# - max_iter=1000: iterasyon sayısını artırdık; bazı veri setlerinde default iterasyon yetersiz kalır
# - model.fit ile eğitim verisi üzerinde parametreler optimize edilir
# ---------------------------------------------------------
model = LogisticRegression(max_iter=1000)  # Logistic Regression sınıflandırıcısı
model.fit(X_train_tfidf, y_train)          # modeli eğitim verisi ile eğit

# ---------------------------------------------------------
# 5) Değerlendirme (Test üzerinde)
# - predict ile test seti için tahmin alınır
# - classification_report: precision/recall/f1 ve support verir (sınıf bazlı performans)
# - accuracy_score: genel doğruluk oranı
# ---------------------------------------------------------
y_pred = model.predict(X_test_tfidf)  # test seti tahminleri

print("\n📊 Model Performansı:")                            # okunabilir çıktı başlığı
print(classification_report(y_test, y_pred))              # detaylı sınıf raporu
print("Genel Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))  # tek satırlık özet

# ---------------------------------------------------------
# 6) Model ve vektörizer kaydetme
# - joblib.dump ile model ve vectorizer'ı diske kaydet; production veya inference için yükleyeceksin
# - kaydetme isimlerini versiyonlayabilir (ör: fault_model_v1.pkl) veya tarihlendirebilirsin
# ---------------------------------------------------------
joblib.dump(model, "fault_model.pkl")        # logistic model kaydı
joblib.dump(vectorizer, "vectorizer.pkl")    # tf-idf vektörizer kaydı
print("\n✅ Model ve vectorizer kaydedildi!")