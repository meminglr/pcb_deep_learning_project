# Uygulama Planı: PCB Anomali Tespiti

## Genel Bakış

Bu plan, PCB anomali tespiti projesini tek bir Jupyter Notebook (`pcb_anomaly_detection.ipynb`) içinde modüler Python sınıfları ve fonksiyonları aracılığıyla adım adım hayata geçirir. Görevler; proje altyapısı, veri katmanı, model tasarımı, eğitim döngüsü, hiperparametre analizi, değerlendirme ve raporlama olmak üzere mantıksal gruplara ayrılmıştır. Her görev bir öncekinin üzerine inşa edilir; hiçbir kod parçası entegre edilmeden bırakılmaz.

## Görevler

- [x] 1. Proje altyapısını oluştur
  - `requirements.txt` dosyasını tasarım belgesindeki paket listesiyle oluştur (`torch>=2.0`, `torchvision>=0.15`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `seaborn>=0.12`, `jupyter>=1.0`, `pandas>=2.0`, `numpy>=1.24`, `Pillow>=9.0`, `ipykernel>=6.0`, `hypothesis>=6.0`)
  - `setup_env.sh` betiğini oluştur: `venv/` mevcutsa yalnızca `pip install -r requirements.txt` çalıştır, değilse `python3 -m venv venv` → `pip install --upgrade pip` → `pip install -r requirements.txt` → `ipykernel install` adımlarını sırayla uygula
  - `.gitignore` dosyasını oluştur: `venv/`, `__pycache__/`, `*.pth`, `.ipynb_checkpoints/` girdilerini ekle
  - _Gereksinimler: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 2. Notebook iskelet yapısını ve Bölüm 1 hücrelerini oluştur
  - [x] 2.1 `pcb_anomaly_detection.ipynb` dosyasını oluştur; 6 ana bölüm için Markdown başlık hücrelerini ekle (`## Bölüm 1: Giriş ve Kurulum` … `## Bölüm 6: Tartışma`); her başlık hücresine bölümün amacını ve beklenen çıktısını açıklayan en az 2 cümle yaz
    - _Gereksinimler: 11.1, 11.2_
  - [x] 2.2 Seed hücresini ekle: `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` — bu hücre tüm import ve işlem hücrelerinden önce gelsin
    - _Gereksinimler: 11.4_
  - [x] 2.3 Import hücresini ve `DATA_ROOT = "dataset/pcb1/Data/Images"` tanım hücresini ekle; cihaz seçim mantığını yaz (MPS → CUDA → CPU önceliği), seçilen cihazı `"Kullanılan cihaz: <cihaz_adı>"` formatında yazdır; GPU yoksa uyarı mesajı göster
    - _Gereksinimler: 5.1, 5.2, 5.4, 11.5_

- [x] 3. `PCBDataset` sınıfını ve `create_dataloaders` fabrika fonksiyonunu uygula
  - [x] 3.1 `PCBDataset` sınıfını Bölüm 2 hücresine yaz: `__init__`, `__len__`, `__getitem__` ve `discover_files` statik metodunu içersin; `Normal/` → etiket 0, `Anomaly/` → etiket 1; bozuk dosyaları `WARNING:` önekiyle logla ve atla; eksik dizinde `FileNotFoundError` fırlat
    - _Gereksinimler: 1.1, 1.8, 1.9_
  - [x]* 3.2 `PCBDataset.discover_files` için property testi yaz (`hypothesis`)
    - **Özellik 1: Dosya Keşfi Etiket Tutarlılığı**
    - **Doğrular: Gereksinim 1.1**
  - [x] 3.3 Eğitim ve doğrulama/test dönüşüm pipeline'larını tanımla (`train_transform`, `val_test_transform`); augmentation: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)`, `ColorJitter(brightness=0.1, contrast=0.1)`, `Resize(224,224)`, `Normalize(ImageNet)`
    - _Gereksinimler: 1.2, 1.3, 1.5, 1.6_
  - [x]* 3.4 `val_test_transform` için property testi yaz
    - **Özellik 2: Görüntü Boyutu Dönüşümü**
    - **Doğrular: Gereksinim 1.2**
  - [x] 3.5 `create_dataloaders` fabrika fonksiyonunu yaz: `train_test_split` ile stratifiye `%70/%15/%15` bölme (seed=42), `WeightedRandomSampler` (replacement=True) ile eğitim DataLoader'ı, sıralı örnekleme ile val/test DataLoader'ları; sınıf ağırlıklarını `[1/n_normal, 1/n_anomaly]` formülüyle hesapla ve döndür; her bölümdeki sınıf dağılımını konsola yazdır
    - _Gereksinimler: 1.4, 1.7, 2.1, 2.2, 2.3, 2.4_
  - [x]* 3.6 Stratifiye bölme için property testi yaz
    - **Özellik 3: Stratifiye Bölme Oranı Korunumu**
    - **Doğrular: Gereksinim 1.4**
  - [x]* 3.7 Sınıf ağırlığı hesaplama için property testi yaz
    - **Özellik 4: Sınıf Ağırlığı Ters Orantı**
    - **Doğrular: Gereksinim 2.2**

- [x] 4. Kontrol noktası — Veri katmanını doğrula
  - Tüm testlerin geçtiğinden emin ol; DataLoader'ların doğru şekilde oluşturulduğunu ve sınıf dağılımlarının konsola yazdırıldığını kontrol et. Sorular varsa kullanıcıya sor.

- [x] 5. Veri analizi ve görselleştirme hücrelerini ekle
  - [x] 5.1 Bölüm 2'ye veri keşfi hücresi ekle: toplam görüntü sayısı, sınıf dağılımı (Normal/Anomaly), örnek görüntüleri (her sınıftan 5 adet) `matplotlib` ile görselleştir
    - _Gereksinimler: 1.7, 11.2_
  - [x]* 5.2 Augmentation pipeline'ının çıktısını gösteren hücre ekle: aynı görüntüye 4 farklı augmentation uygulanmış hallerini yan yana göster
    - _Gereksinimler: 1.5_

- [x] 6. `CustomCNN` sınıfını uygula
  - [x] 6.1 `CustomCNN` sınıfını Bölüm 3 hücresine yaz: 3 evrişimli blok (`Conv2d → BatchNorm2d → ReLU → MaxPool2d`, filtreler: 32→64→128), `AdaptiveAvgPool2d(4,4)`, `Flatten`, `Linear(2048,256)`, `ReLU`, `Dropout(0.5)`, `Linear(256,2)`; `forward` ve `count_parameters` metodlarını ekle; parametre sayısını notebook çıktısında raporla
    - _Gereksinimler: 3.1, 3.2, 3.3, 3.5_
  - [ ]* 6.2 `CustomCNN` çıktı şekli için property testi yaz
    - **Özellik 5: CNN Çıktı Şekli**
    - **Doğrular: Gereksinim 3.3, 3.4**

- [x] 7. `TransferModel` sınıfını uygula
  - [x] 7.1 `TransferModel` sınıfını Bölüm 3 hücresine yaz: ResNet-18 omurgasını `weights=ResNet18_Weights.IMAGENET1K_V1` ile yükle; `conv1`, `bn1`, `relu`, `maxpool`, `layer1`, `layer2` katmanlarını `requires_grad=False` ile dondur; `layer3`, `layer4` eğitilebilir bırak; orijinal `fc` başlığını `Linear(512,256) → ReLU → Dropout(0.5) → Linear(256,2)` ile değiştir; `freeze_backbone` ve `count_parameters` metodlarını ekle; parametre sayısını raporla
    - _Gereksinimler: 4.1, 4.2, 4.3, 4.4, 4.5_
  - [ ]* 7.2 `TransferModel.freeze_backbone` için property testi yaz
    - **Özellik 6: Transfer Model Katman Dondurma Tutarlılığı**
    - **Doğrular: Gereksinim 4.2**
  - [ ]* 7.3 `TransferModel` çıktı şekli için property testi yaz
    - **Özellik 7: Transfer Model Çıktı Şekli**
    - **Doğrular: Gereksinim 4.4**

- [x] 8. Kontrol noktası — Model mimarilerini doğrula
  - Tüm testlerin geçtiğinden emin ol; her iki modelin `(batch_size, 2)` çıktı ürettiğini ve parametre sayılarının raporlandığını kontrol et. Sorular varsa kullanıcıya sor.

- [x] 9. `Trainer` sınıfını uygula
  - [x] 9.1 `Trainer` sınıfını Bölüm 4 hücresine yaz: `__init__` (model, device, class_weights, lr, max_epochs, patience), `_train_epoch`, `_validate_epoch`, `_save_best` ve `train` metodlarını içersin; `Adam` optimizer (`filter(lambda p: p.requires_grad, ...)`), `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` scheduler, `CrossEntropyLoss(weight=class_weights)` kayıp fonksiyonu; her epoch sonunda loss/accuracy değerlerini konsola yazdır ve `history` dict'e kaydet; early stopping (patience=10) tetiklendiğinde epoch numarasını yazdır; eğitim süresini `"Eğitim süresi: X dakika Y saniye"` formatında raporla
    - _Gereksinimler: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [ ]* 9.2 `Trainer.train` eğitim geçmişi eksiksizliği için property testi yaz
    - **Özellik 8: Eğitim Geçmişi Eksiksizliği**
    - **Doğrular: Gereksinim 6.1**
  - [ ]* 9.3 Early stopping tetiklenme garantisi için property testi yaz
    - **Özellik 9: Early Stopping Tetiklenme Garantisi**
    - **Doğrular: Gereksinim 6.2**
  - [ ]* 9.4 En iyi model kaydetme doğruluğu için property testi yaz
    - **Özellik 10: En İyi Model Kaydetme Doğruluğu**
    - **Doğrular: Gereksinim 6.3**

- [x] 10. CNN ve Transfer Learning modellerini eğit
  - [x] 10.1 Bölüm 4'e CNN eğitim hücresini ekle: `Trainer` örneği oluştur (lr=1e-3, max_epochs=50, patience=10), `trainer.train(train_loader, val_loader, "cnn")` çağır, `cnn_history` değişkenine kaydet
    - _Gereksinimler: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 10.2 Bölüm 4'e Transfer Learning eğitim hücresini ekle: `TransferModel` örneği oluştur, `freeze_backbone()` çağır, `Trainer` örneği oluştur (lr=1e-3, max_epochs=50, patience=10), `trainer.train(train_loader, val_loader, "transfer")` çağır, `transfer_history` değişkenine kaydet
    - _Gereksinimler: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 11. `HyperparamAnalyzer` sınıfını uygula ve LR analizini çalıştır
  - [x] 11.1 `HyperparamAnalyzer` sınıfını Bölüm 4 hücresine yaz: `run_lr_search` metodu `lr_values=[1e-2, 1e-3, 1e-4]` için her deneyde `torch.manual_seed(42)` + `numpy.random.seed(42)` ayarla, yeni `CustomCNN` örneği oluştur, `Trainer(max_epochs=20, patience=5)` ile eğit, `val_f1_weighted` hesapla; `plot_lr_analysis` metodu log-scale x ekseniyle çizgi grafik çizsin; en iyi LR'yi `"En iyi öğrenme oranı: <değer>, Doğrulama F1: <skor>"` formatında yazdır
    - _Gereksinimler: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [ ]* 11.2 `HyperparamAnalyzer.run_lr_search` sonuç eksiksizliği için property testi yaz
    - **Özellik 11: Hiperparametre Analizi Sonuç Eksiksizliği**
    - **Doğrular: Gereksinim 7.2**
  - [x] 11.3 LR analizi hücresini ekle: `HyperparamAnalyzer` örneği oluştur, `run_lr_search()` çağır, sonuçları DataFrame olarak göster, `plot_lr_analysis()` çağır
    - _Gereksinimler: 7.3, 7.4_

- [x] 12. Kontrol noktası — Eğitim ve analiz aşamasını doğrula
  - Tüm testlerin geçtiğinden emin ol; `best_cnn.pth` ve `best_transfer.pth` dosyalarının oluşturulduğunu, LR analiz tablosunun 3 satır içerdiğini kontrol et. Sorular varsa kullanıcıya sor.

- [x] 13. `Evaluator` sınıfını uygula
  - [x] 13.1 `Evaluator` sınıfını Bölüm 5 hücresine yaz: `evaluate` metodu `best_<model_adı>.pth` ağırlıklarını yüklesin (dosya yoksa `FileNotFoundError`), test seti üzerinde `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `f1_weighted`, `roc_auc`, `avg_precision` metriklerini `sklearn.metrics` ile hesaplayıp döndürsün; `plot_confusion_matrix` (seaborn heatmap, sayısal değerler), `plot_roc_curves` (her iki model aynı grafik, referans çizgisi), `plot_pr_curves` (her iki model aynı grafik), `plot_misclassified` (FP ve FN ayrı ayrı, en az 5 örnek, üzerinde gerçek/tahmin etiketi), `plot_training_history` (loss ve accuracy grafikleri) metodlarını ekle
    - _Gereksinimler: 9.1, 9.2, 9.3, 9.4, 9.6, 10.1_
  - [ ]* 13.2 `Evaluator.evaluate` metrik kapsamlılığı için property testi yaz
    - **Özellik 14: Metrik Hesaplama Kapsamlılığı**
    - **Doğrular: Gereksinim 9.1**
  - [ ]* 13.3 ROC-AUC hesaplama doğruluğu için property testi yaz
    - **Özellik 15: ROC-AUC Hesaplama Doğruluğu**
    - **Doğrular: Gereksinim 9.3**

- [x] 14. Overfitting analizi hücrelerini ekle
  - [x] 14.1 Bölüm 5'e overfitting analiz hücrelerini ekle: `plot_training_history` ile her model için loss ve accuracy eğrilerini çiz; son 5 epoch ortalamasıyla overfitting gap'i `"Overfitting gap (son 5 epoch ortalaması): <değer>"` formatında raporla; `train_acc[i] - val_acc[i] > 10.0` koşulunu sağlayan epoch'ları dikey kesikli çizgiyle işaretle (yoksa `"Overfitting eşiği aşılmadı"` yazdır); tartışma bölümü için overfitting gap, early stopping durumu ve Dropout katmanlarını listele
    - _Gereksinimler: 8.1, 8.2, 8.3, 8.4, 8.5_
  - [ ]* 14.2 Overfitting gap hesaplama doğruluğu için property testi yaz
    - **Özellik 12: Overfitting Gap Hesaplama Doğruluğu**
    - **Doğrular: Gereksinim 8.3**
  - [ ]* 14.3 Overfitting eşiği tespiti için property testi yaz
    - **Özellik 13: Overfitting Eşiği Tespiti**
    - **Doğrular: Gereksinim 8.4**

- [x] 15. Model karşılaştırma tablosu ve tartışma hücrelerini ekle
  - [x] 15.1 Bölüm 5'e karşılaştırma tablosu hücresini ekle: `Evaluator` örnekleri oluştur, her iki modeli değerlendir, `Accuracy`, `Macro-F1`, `Weighted-F1`, `ROC-AUC`, `AP` metriklerini içeren pandas DataFrame tablosu göster; `plot_roc_curves` ve `plot_pr_curves` ile her iki modelin eğrilerini aynı grafik üzerinde çiz
    - _Gereksinimler: 9.3, 9.4, 9.5_
  - [x] 15.2 Bölüm 5'e detaylı karşılaştırma tablosu hücresini ekle: Model Adı, Toplam Parametre, Eğitilebilir Parametre, Eğitim Süresi (dk), Test Macro-F1, Test ROC-AUC sütunlarını içeren DataFrame; `plot_misclassified` ile her iki model için FP/FN görselleştirmesi
    - _Gereksinimler: 10.1, 10.2_
  - [x] 15.3 Bölüm 6'ya model öneri mantığı hücresini ekle: F1 farkı ≥ 0.05 ise daha yüksek F1'li modeli, fark < 0.05 ise daha az parametreli modeli öner; ΔPrecision ve ΔRecall değerlerini hesaplayıp sınıf dengesizliğinin etkisini sayısal olarak göster
    - _Gereksinimler: 10.3, 10.4_
  - [ ]* 15.4 Model öneri mantığı doğruluğu için property testi yaz
    - **Özellik 16: Model Öneri Mantığı Doğruluğu**
    - **Doğrular: Gereksinim 10.3**

- [ ] 16. Property-based test dosyasını oluştur
  - [ ] 16.1 `tests/test_properties.py` dosyasını oluştur: `hypothesis` kütüphanesiyle tüm 16 özellik için `@settings(max_examples=100)` dekoratörlü test fonksiyonlarını yaz; her test `# Feature: pcb-anomaly-detection, Property {N}: {özellik_metni}` formatında etiketlensin; test dosyası notebook'tan bağımsız olarak `pytest tests/test_properties.py` komutuyla çalıştırılabilsin
    - _Gereksinimler: Tüm özellikler (1–16)_
  - [ ]* 16.2 `tests/test_units.py` dosyasını oluştur: kenar durumlar için birim testler yaz (bozuk görüntü, eksik dizin, GPU yokluğu, augmentation pipeline varlığı, scheduler parametreleri, DataFrame sütunları)
    - _Gereksinimler: 1.8, 1.9, 5.4, 9.6_

- [x] 17. `README.md` dosyasını oluştur
  - Proje amacını ve SPOT-Diff veri setini `[SPOT-Diff](https://github.com/amazon-science/spot-diff)` bağlantısıyla açıkla; `## Kurulum` bölümüne 4 adımlı kurulum talimatlarını yaz (klonla → `bash setup_env.sh` → `source venv/bin/activate` → `jupyter notebook pcb_anomaly_detection.ipynb`); dizin yapısını ağaç diyagramıyla göster; `## Sonuçlar` bölümüne Macro-F1 ve ROC-AUC için yer tutucu değerli Markdown tablosu ekle
  - _Gereksinimler: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 18. Son kontrol noktası — Tüm testlerin geçtiğini doğrula
  - `pytest tests/` komutuyla tüm testleri çalıştır; notebook'u baştan sona sırayla çalıştırarak herhangi bir `Exception` veya `Error` üretmediğini doğrula. Sorular varsa kullanıcıya sor.

## Notlar

- `*` ile işaretli görevler isteğe bağlıdır; MVP için atlanabilir
- Her görev belirli gereksinimlere referans verir; izlenebilirlik sağlanmıştır
- Kontrol noktaları artımlı doğrulama sağlar
- Property testleri evrensel doğruluk özelliklerini, birim testler ise belirli örnekleri ve kenar durumları doğrular
- `num_workers=0` MPS uyumluluğu için zorunludur
- Tüm eğitim deneyleri `torch.manual_seed(42)` ile tekrarlanabilirlik garantisi altındadır

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "2.2", "2.3"] },
    { "id": 1, "tasks": ["3.1", "3.3"] },
    { "id": 2, "tasks": ["3.2", "3.4", "3.5"] },
    { "id": 3, "tasks": ["3.6", "3.7", "5.1", "5.2"] },
    { "id": 4, "tasks": ["6.1", "7.1"] },
    { "id": 5, "tasks": ["6.2", "7.2", "7.3"] },
    { "id": 6, "tasks": ["9.1"] },
    { "id": 7, "tasks": ["9.2", "9.3", "9.4", "10.1", "10.2"] },
    { "id": 8, "tasks": ["11.1"] },
    { "id": 9, "tasks": ["11.2", "11.3"] },
    { "id": 10, "tasks": ["13.1"] },
    { "id": 11, "tasks": ["13.2", "13.3", "14.1"] },
    { "id": 12, "tasks": ["14.2", "14.3", "15.1"] },
    { "id": 13, "tasks": ["15.2", "15.3"] },
    { "id": 14, "tasks": ["15.4", "16.1"] },
    { "id": 15, "tasks": ["16.2"] }
  ]
}
```
