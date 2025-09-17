# image_inference.py
# Bu dosya araç hasar tespiti için YOLOv8 modelini kullanır.
# Görsellerde tespit edilen hasarları JSON formatında döner ve işaretlenmiş görseli kaydeder.

from ultralytics import YOLO        # YOLOv8 model sınıfı (ultralytics kütüphanesinden)
import cv2                          # OpenCV: görselleri işlemek, kutu çizmek, kaydetmek için
import os                           # Dosya/dizin işlemleri için
from datetime import datetime       # Annotated dosya ismine timestamp eklemek için

# -----------------------------------------------------
# Eğitimli YOLOv8 modelini yükle
# (custom olarak finetune edilmiş ağırlık dosyası)
# -----------------------------------------------------
yolo_model = YOLO("models/car_damage_yolov8.pt")  # Model dosya yolu (önceden eğitilmiş ağırlıklar)

# -----------------------------------------------------
# Tespit edilen sınıflara açıklama ve öneriler eşleştirilir
# Bu dictionary, modelin döndüğü class label'larını Türkçe açıklama + öneri ile map eder
# -----------------------------------------------------
CLASS_EXPLANATIONS = {
    "hose_tear": ("Hortum yırtığı", "Hortumun değiştirilmesi/yenilenmesi gerekir."),
    "loose_cable": ("Çıkan kablo", "Kablo monte edilmeli / bağlantı noktası kontrol edilmeli."),
    # Diğer sınıfları buraya ekleyebilirsin (ör: broken_light, scratch, rust vs.)
}

# -----------------------------------------------------
# Fonksiyon: run_image_detection
# Parametre: file_path -> giriş görselinin yolu
# Çıkış: (mapped_detections, annotated_path)
#   mapped_detections: [{label, conf, bbox, title, advice}, ...]
#   annotated_path: işaretlenmiş görselin diskteki yolu
# -----------------------------------------------------
def run_image_detection(file_path):
    # 1) YOLOv8 ile prediction yap
    # source: görsel yolu
    # imgsz: giriş resim boyutu (640 önerilen default)
    # conf: minimum güven eşiği
    # verbose=False: konsolda gereksiz log yazdırmayı kapatır
    results = yolo_model.predict(source=file_path, imgsz=640, conf=0.25, verbose=False)[0]

    detections = []  # ham tespitleri saklayacağımız liste

    # 2) Görseli OpenCV ile yükle (kutular üzerine çizim yapabilmek için)
    img = cv2.imread(file_path)

    # 3) Modelden gelen her box için:
    # results.boxes.xyxy -> [x1, y1, x2, y2] koordinatları
    # results.boxes.cls -> class id
    # results.boxes.conf -> güven skoru
    for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        # Kutunun koordinatlarını int'e çevir
        x1, y1, x2, y2 = map(int, box.tolist())
        label = yolo_model.names[int(cls_id)]   # class id → class label (ör: "hose_tear")
        conf = float(score)                     # güven skoru

        # Ham tespiti listeye ekle
        detections.append({
            "label": label,
            "conf": conf,
            "bbox": [x1, y1, x2, y2]
        })

        # 4) Annotated görsel üzerine kutu çiz
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)  # yeşil kutu
        cv2.putText(
            img,
            f"{label} {conf:.2f}",   # label + güven skoru
            (x1, y1-10),             # kutunun üstünde göster
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # 5) Annotated görseli kaydet
    annotated_dir = "uploads/annotated"
    os.makedirs(annotated_dir, exist_ok=True)   # klasör yoksa oluştur
    annotated_path = os.path.join(
        annotated_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S_") + os.path.basename(file_path)  # timestamp + orijinal dosya adı
    )
    cv2.imwrite(annotated_path, img)  # Annotated resmi kaydet

    # 6) Tespitleri açıklama/öneri ile eşle
    mapped = []
    for d in detections:
        # Eğer label sözlükte varsa title+advice al, yoksa "Öneri yok"
        title, advice = CLASS_EXPLANATIONS.get(d["label"], (d["label"], "Öneri yok"))
        mapped.append({**d, "title": title, "advice": advice})

    # 7) JSON için mapped tespitleri ve annotated resim yolunu döndür
    return mapped, annotated_path
