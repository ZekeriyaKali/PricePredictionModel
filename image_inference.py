from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# pretrained custom weights (finetuned)
yolo_model = YOLO("models/car_damage_yolov8.pt")  # path to your trained weights

CLASS_EXPLANATIONS = {
    "hose_tear": ("Hortum yırtığı", "Hortumun değiştirilmesi/yenilenmesi gerekir."),
    "loose_cable": ("Çıkan kablo", "Kablo monte edilmeli / bağlantı noktası kontrol edilmeli."),
    # ... diğer sınıflar
}

def run_image_detection(file_path):
    results = yolo_model.predict(source=file_path, imgsz=640, conf=0.25, verbose=False)[0]
    detections = []
    img = cv2.imread(file_path)
    for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = yolo_model.names[int(cls_id)]
        conf = float(score)
        detections.append({"label": label, "conf": conf, "bbox": [x1, y1, x2, y2]})
        # draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # save annotated image
    annotated_dir = "uploads/annotated"
    os.makedirs(annotated_dir, exist_ok=True)
    annotated_path = os.path.join(annotated_dir, datetime.now().strftime("%Y%m%d_%H%M%S_") + os.path.basename(file_path))
    cv2.imwrite(annotated_path, img)

    # map explanations
    mapped = []
    for d in detections:
        title, advice = CLASS_EXPLANATIONS.get(d["label"], (d["label"], "Öneri yok"))
        mapped.append({**d, "title": title, "advice": advice})

    return mapped, annotated_path