import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Veriyi yükle
df = pd.read_csv("car_faults.csv", delimiter=";")
print("Veri örneği:")
print(df.head())
print("Kolonlar:", df.columns)

# 2. Train-test ayır
X_train, X_test, y_train, y_test = train_test_split(
    df["description"], df["issue"], test_size=0.2, random_state=42
)

# 3. Metni sayısallaştır (TF-IDF + n-gram)
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Model eğit (daha güçlü Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Test et
y_pred = model.predict(X_test_tfidf)

print("\n📊 Model Performansı:")
print(classification_report(y_test, y_pred))
print("Genel Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))

# 6. Kaydet
joblib.dump(model, "fault_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ Model ve vectorizer kaydedildi!")