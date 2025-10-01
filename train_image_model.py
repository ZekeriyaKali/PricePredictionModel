# train_image_model.py
# Bu script: image_dataset klasöründeki resimleri kullanarak
# ResNet18 tabanlı transfer learning modeli eğitir ve kaydeder.

import torch  # PyTorch ana kütüphanesi
import torch.nn as nn  # Neural network katmanları (Linear, Conv, Loss vb.)
import torch.optim as optim  # Optimizasyon algoritmaları (Adam, SGD vb.)
from torchvision import datasets, models, transforms  # Görüntü dataset, hazır modeller, transform işlemleri
from torch.utils.data import DataLoader  # Mini-batch data loader için

# -------------------------
# Dataset klasör yapısı (beklenen):
# image_dataset/
#   normal/
#   kablo_cikmis/
#   hortum_delik/
# -------------------------

DATASET_PATH = "image_detect/images"  # Eğitim verisinin bulunduğu klasör
BATCH_SIZE = 16  # Her iterasyonda işlenecek örnek sayısı
EPOCHS = 5  # Eğitimde kaç epoch (tüm veri seti üzerinden geçiş) yapılacağı
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Eğer GPU varsa "cuda" seçilir, yoksa CPU kullanılır.

# -------------------------
# Görüntü Transformları (ön işleme + augmentation)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resimleri 224x224 boyutuna ölçekle (ResNet input boyutu)
    transforms.ToTensor(),  # [0,255] arası pikselleri [0,1] float tensora çevir
    transforms.Normalize(  # Normalizasyon (ImageNet ortalama/std ile)
        [0.485, 0.456, 0.406],  # mean
        [0.229, 0.224, 0.225]  # std
    )
])

# -------------------------
# Dataset ve DataLoader
# -------------------------
# datasets.ImageFolder: Alt klasör isimlerini otomatik sınıf etiketi olarak alır
train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

# DataLoader: batch halinde veriyi yükler, shuffle=True => her epoch rastgele sıralama
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# Model Tanımlama (ResNet18 Transfer Learning)
# -------------------------
# torchvision.models.resnet18(pretrained=True): ImageNet üzerinde önceden eğitilmiş ağırlıklarla başlar
model = models.resnet18(pretrained=True)

# Çıkış sınıfı sayısını dataset’e göre ayarlama:
num_classes = len(train_dataset.classes)  # Örn: 3 (normal, kablo_cikmis, hortum_delik)
# ResNet’in son katmanını (fc: fully connected layer) kendi sınıf sayımıza göre değiştiriyoruz
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Modeli uygun cihaza (GPU/CPU) gönder
model = model.to(DEVICE)

# -------------------------
# Loss Fonksiyonu & Optimizatör
# -------------------------
criterion = nn.CrossEntropyLoss()  # Çok sınıflı sınıflandırma için uygun loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer, öğrenme oranı=0.001

# -------------------------
# Eğitim Döngüsü
# -------------------------
for epoch in range(EPOCHS):
    model.train()  # Eğitim moduna al (dropout, batchnorm aktifleşir)
    running_loss = 0.0  # Her epoch’taki toplam loss’u izlemek için

    # Mini-batch döngüsü
    for inputs, labels in train_loader:
        # Verileri GPU/CPU’ya gönder
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Optimizatörün gradientlerini sıfırla (önceki batch’den kalanlar temizlenmeli)
        optimizer.zero_grad()

        # İleri yayılım (forward pass)
        outputs = model(inputs)

        # Loss hesapla
        loss = criterion(outputs, labels)

        # Geri yayılım (backpropagation)
        loss.backward()

        # Ağırlıkları güncelle
        optimizer.step()

        # Loss’u topla
        running_loss += loss.item()

    # Epoch sonunda ortalama loss yazdır
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}")

# -------------------------
# Modeli kaydet
# -------------------------
# state_dict(): modelin ağırlıklarını (parametrelerini) içerir
torch.save(model.state_dict(), "image_model.pth")
print("✅ Image model kaydedildi: image_model.pth")