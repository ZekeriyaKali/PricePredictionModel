from ultralytics import YOLO

# data.yaml içeriği:
# train: /path/to/train/images
# val: /path/to/val/images
# nc: <num_classes>
# names: ['hose_tear', 'loose_cable', 'crack', ...]

model = YOLO("yolov8n.pt")   # küçük model başlangıç
model.train(data="data.yaml", epochs=100, imgsz=640, batch=16, name="car_damage_yolov8")