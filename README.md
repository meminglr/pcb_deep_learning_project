# PCB Anomali Tespiti — Derin Öğrenme Projesi

Bu proje, [SPOT-Diff](https://github.com/amazon-science/spot-diff) veri setinden alınan PCB (Baskılı Devre Kartı) görüntüleri üzerinde derin öğrenme tabanlı ikili sınıflandırma (Normal / Anomaly) gerçekleştirmektedir. İki farklı mimari karşılaştırılmaktadır: sıfırdan tasarlanmış özel bir CNN ve ResNet-18 tabanlı Transfer Learning modeli.

---

## Proje Yapısı

```
derin_ogrenme_pcb/
├── pcb_anomaly_detection.ipynb   ← Ana Jupyter Notebook
├── requirements.txt               ← Python bağımlılıkları
├── setup_env.sh                   ← Sanal ortam kurulum betiği
├── README.md                      ← Bu dosya
├── .gitignore
├── best_cnn.pth                   ← En iyi CNN ağırlıkları (eğitim sonrası)
├── best_transfer.pth              ← En iyi Transfer ağırlıkları (eğitim sonrası)
└── dataset/
    └── pcb1/
        └── Data/
            └── Images/
                ├── Normal/        ← ~1004 .JPG (etiket: 0)
                └── Anomaly/       ← ~100 .JPG  (etiket: 1)
```

---

## Kurulum

### 1. Repoyu klonla

```bash
git clone <repo-url>
cd derin_ogrenme_pcb
```

### 2. Sanal ortamı kur

```bash
bash setup_env.sh
```

Bu betik:
- `venv/` sanal ortamını oluşturur (zaten varsa atlar)
- Tüm bağımlılıkları yükler (`requirements.txt`)
- Jupyter çekirdeğini `Python (venv)` olarak kaydeder

### 3. Sanal ortamı etkinleştir

```bash
source venv/bin/activate
```

### 4. Notebook'u aç

```bash
jupyter notebook pcb_anomaly_detection.ipynb
```

Notebook açıldığında çekirdek olarak **Python (venv)** seçin.

---

## Veri Seti

Veri seti `dataset/pcb1/Data/Images/` dizininde bulunmalıdır:

```
dataset/
└── pcb1/
    └── Data/
        └── Images/
            ├── Normal/     ← ~1004 .JPG görüntü
            └── Anomaly/    ← ~100 .JPG görüntü
```

Veri seti kaynağı: [SPOT-Diff GitHub](https://github.com/amazon-science/spot-diff)

---

## Notebook İçeriği

| Bölüm | İçerik |
|---|---|
| Bölüm 1 | Giriş, kurulum, seed ayarları, cihaz seçimi (MPS/CUDA/CPU) |
| Bölüm 2 | Veri analizi, sınıf dağılımı, augmentation pipeline, DataLoader |
| Bölüm 3 | CustomCNN ve TransferModel (ResNet-18) mimarileri |
| Bölüm 4 | Model eğitimi, hiperparametre analizi (LR), early stopping |
| Bölüm 5 | Performans metrikleri, ROC/PR eğrileri, karşılaştırma tablosu |
| Bölüm 6 | Overfitting analizi, model önerisi, tartışma |

---

## Gereksinimler

```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
pandas>=2.0
numpy>=1.24
Pillow>=9.0
ipykernel>=6.0
hypothesis>=6.0
```

---

## Platform

Apple Silicon M2 MacBook Air — MPS backend ile GPU hızlandırması sağlanmaktadır.

Cihaz seçimi otomatik yapılır: **MPS → CUDA → CPU**

---

## Sonuçlar

Aşağıdaki tablo eğitim tamamlandıktan sonra doldurulacaktır:

| Model | Test Macro-F1 | Test ROC-AUC | Eğitim Süresi (dk) | Parametre Sayısı |
|---|---|---|---|---|
| CustomCNN | — | — | — | ~550K |
| TransferModel (ResNet-18) | — | — | — | ~11.2M toplam / ~4.2M eğitilebilir |

---

## Lisans

Bu proje eğitim amaçlıdır. Veri seti için [SPOT-Diff lisansına](https://github.com/amazon-science/spot-diff) bakınız.
# pcb_deep_learning_project
