# Gereksinimler Belgesi

## Giriş

Bu proje, SPOT-Diff veri setinden alınan PCB (Baskılı Devre Kartı) görüntüleri üzerinde derin öğrenme tabanlı anomali tespiti gerçekleştirmeyi amaçlamaktadır. Veri seti, `dataset/pcb1/Data/Images/` dizininde yer alan Normal (~1004 görüntü) ve Anomaly (~100 görüntü) sınıflarından oluşmaktadır. Proje, Apple Silicon M2 MacBook Air üzerinde MPS backend desteğiyle PyTorch kullanılarak Jupyter Notebook formatında geliştirilecektir. En az iki mimari olarak farklı derin öğrenme modeli (özel CNN ve Transfer Learning) karşılaştırmalı olarak değerlendirilecektir.

---

## Sözlük

- **Sistem**: PCB anomali tespiti için geliştirilen Jupyter Notebook tabanlı derin öğrenme sistemi bütünü.
- **Veri_Yukleyici**: PyTorch `DataLoader` ve `Dataset` sınıflarını kullanarak görüntüleri diskten okuyup ön işleme tabi tutan modül.
- **CNN_Modeli**: Sıfırdan tasarlanmış, evrişimli katmanlardan oluşan özel sinir ağı mimarisi.
- **Transfer_Modeli**: ImageNet ağırlıklarıyla önceden eğitilmiş bir omurga ağı (ResNet-18 veya EfficientNet-B0) üzerine inşa edilen ince ayar (fine-tuning) modeli.
- **Egitici**: Eğitim döngüsünü, kayıp hesaplamayı ve ağırlık güncellemelerini yöneten modül.
- **Degerlendiricı**: Test seti üzerinde metrikleri hesaplayan ve görselleştiren modül.
- **Hiperparametre_Analizci**: Farklı hiperparametre kombinasyonlarını sistematik biçimde deneyen ve sonuçları kaydeden modül.
- **Normal**: Üretim hatası içermeyen, referans PCB görüntüsü sınıfı (etiket: 0).
- **Anomaly**: Üretim hatası veya kusur içeren PCB görüntüsü sınıfı (etiket: 1).
- **MPS**: Apple Silicon işlemcilerde GPU hızlandırması sağlayan Metal Performance Shaders backend.
- **Sinif_Dengesizligi**: Normal sınıfının (~1004) Anomaly sınıfına (~100) oranla yaklaşık 10:1 oranında fazla olması durumu.
- **Overfitting**: Modelin eğitim verisini ezberleyip doğrulama/test verisinde genelleme yapamaması durumu.
- **ROC_AUC**: Receiver Operating Characteristic eğrisi altında kalan alan; ikili sınıflandırma performansının eşik bağımsız ölçütü.
- **Karisiklik_Matrisi**: Gerçek ve tahmin edilen sınıf etiketlerini karşılaştıran tablo (TP, TN, FP, FN).

---

## Gereksinimler

### Gereksinim 1: Veri Yükleme ve Ön İşleme

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, PCB görüntülerini tutarlı biçimde yükleyip ön işlemden geçirmek istiyorum; böylece modeller aynı kalitede girdi alır.

#### Kabul Kriterleri

1. THE Veri_Yukleyici SHALL `dataset/pcb1/Data/Images/Normal/` ve `dataset/pcb1/Data/Images/Anomaly/` dizinlerindeki tüm `.JPG` dosyalarını otomatik olarak keşfedip etiketlendirerek yüklesin; Normal sınıfına 0, Anomaly sınıfına 1 etiketi atansın.
2. THE Veri_Yukleyici SHALL tüm görüntüleri 224×224 piksel boyutuna yeniden ölçeklendirsin (bilinear interpolasyon kullanılsın).
3. THE Veri_Yukleyici SHALL her görüntüyü ImageNet istatistikleriyle (ortalama=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) normalize etsin.
4. THE Veri_Yukleyici SHALL veri setini tekrarlanabilir bir rastgele tohum (random seed = 42) kullanarak %70 eğitim, %15 doğrulama, %15 test olarak böleceği şekilde `sklearn.model_selection.train_test_split` ile stratifiye örnekleme uygulasın; her bölümde Normal/Anomaly oranı korunsun.
5. WHILE eğitim verisi işlenirken, THE Veri_Yukleyici SHALL yatay çevirme (p=0.5), ±15° döndürme ve ±%10 parlaklık/kontrast değişimi içeren veri artırma (data augmentation) dönüşümlerini uygulasın.
6. WHILE doğrulama ve test verisi işlenirken, THE Veri_Yukleyici SHALL yalnızca yeniden ölçeklendirme ve normalizasyon uygulasın; veri artırma uygulamasın.
7. WHEN veri seti yüklemesi tamamlandığında, THE Veri_Yukleyici SHALL eğitim, doğrulama ve test bölümlerinin her birindeki Normal ve Anomaly görüntü sayılarını ve oranlarını konsola yazdırsın.
8. IF bir görüntü dosyası bozuk veya okunamaz durumdaysa, THEN THE Veri_Yukleyici SHALL söz konusu dosyanın tam yolunu ve hata mesajını konsola `WARNING:` önekiyle yazdırsın ve kalan görüntülerle işleme devam etsin.
9. IF `dataset/pcb1/Data/Images/Normal/` veya `dataset/pcb1/Data/Images/Anomaly/` dizinlerinden biri mevcut değilse, THEN THE Veri_Yukleyici SHALL `FileNotFoundError` fırlatarak hangi dizinin eksik olduğunu belirten bir hata mesajı göstersin.

---

### Gereksinim 2: Sınıf Dengesizliği Yönetimi

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, ~10:1 sınıf dengesizliğini ele almak istiyorum; böylece modeller azınlık sınıfını (Anomaly) görmezden gelmez.

#### Kabul Kriterleri

1. THE Egitici SHALL eğitim DataLoader'ında `torch.utils.data.WeightedRandomSampler` kullanarak her mini-batch'te Normal ve Anomaly örneklerini yaklaşık 1:1 oranında örneklesin; `replacement=True` parametresi etkin olsun.
2. THE Egitici SHALL `CrossEntropyLoss` kayıp fonksiyonunda `weight` parametresini `[1/n_normal, 1/n_anomaly]` formülüyle hesaplayarak sınıf frekanslarının tersine orantılı biçimde ayarlasın.
3. WHEN sınıf ağırlıkları hesaplanırken, THE Egitici SHALL ağırlıkları yalnızca eğitim bölümündeki sınıf frekanslarından türetsin; doğrulama veya test verisindeki örnekleri hesaba katmasın.
4. THE Egitici SHALL hesaplanan sınıf ağırlıklarını (Normal ağırlığı ve Anomaly ağırlığı) notebook çıktısında sayısal olarak göstersin.

---

### Gereksinim 3: CNN Modeli Tasarımı

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, sıfırdan tasarlanmış bir CNN modeli geliştirmek istiyorum; böylece temel bir derin öğrenme mimarisinin PCB anomali tespitindeki performansını ölçebilirim.

#### Kabul Kriterleri

1. THE CNN_Modeli SHALL en az 3 evrişimli blok içersin; her blok sırasıyla `Conv2d`, `BatchNorm2d`, `ReLU` ve `MaxPool2d` katmanlarından oluşsun; filtre sayıları bloktan bloğa artacak şekilde tasarlansın (örn. 32→64→128).
2. THE CNN_Modeli SHALL tam bağlantılı sınıflandırıcı başlığında 0.3 ≤ oran ≤ 0.6 aralığında en az bir `Dropout` katmanı içersin.
3. THE CNN_Modeli SHALL 2 nöronlu logit çıktısı üretsin (indeks 0: Normal, indeks 1: Anomaly); `softmax` veya `sigmoid` aktivasyonu modelin içinde uygulanmasın, kayıp fonksiyonuna ham logit geçilsin.
4. WHEN 224×224×3 boyutunda bir girdi tensörü modele verildiğinde, THE CNN_Modeli SHALL herhangi bir `RuntimeError` veya boyut uyuşmazlığı hatası üretmeden (batch_size, 2) boyutunda bir çıktı tensörü döndürsün.
5. WHEN model oluşturulduğunda, THE CNN_Modeli SHALL toplam parametre sayısını ve eğitilebilir parametre sayısını ayrı ayrı notebook hücresinde raporlasın.

---

### Gereksinim 4: Transfer Learning Modeli Tasarımı

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, önceden eğitilmiş bir model üzerine ince ayar uygulamak istiyorum; böylece sınırlı veriyle daha yüksek performans elde edip CNN ile karşılaştırabilirim.

#### Kabul Kriterleri

1. THE Transfer_Modeli SHALL `torchvision.models` üzerinden ImageNet ağırlıklarıyla önceden eğitilmiş ResNet-18 veya EfficientNet-B0 omurga ağını temel alsın; ağırlıklar `weights=...IMAGENET1K_V1` parametresiyle yüklensin.
2. THE Transfer_Modeli SHALL omurga ağının son iki katman bloğunu (`layer3` ve `layer4` ResNet-18 için; son iki `MBConv` bloğu EfficientNet-B0 için) eğitilebilir bırakırken önceki tüm katmanları `requires_grad=False` ile dondursun.
3. THE Transfer_Modeli SHALL orijinal sınıflandırıcı başlığını; `Linear → ReLU → Dropout(0.5) → Linear(2)` yapısında yeni bir tam bağlantılı başlıkla değiştirsin.
4. THE Transfer_Modeli SHALL 2 nöronlu logit çıktısı üretsin (indeks 0: Normal, indeks 1: Anomaly); ham logit kayıp fonksiyonuna geçilsin.
5. WHEN model oluşturulduğunda, THE Transfer_Modeli SHALL toplam parametre sayısını ve eğitilebilir (dondurulmamış) parametre sayısını ayrı ayrı notebook hücresinde raporlasın.

---

### Gereksinim 5: Cihaz Yönetimi

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, Apple Silicon M2 MacBook Air üzerinde MPS hızlandırmasını kullanmak istiyorum; böylece eğitim süresi makul düzeyde kalır.

#### Kabul Kriterleri

1. THE Sistem SHALL aşağıdaki öncelik sırasına göre hesaplama cihazını otomatik seçsin: (1) `torch.backends.mps.is_available()` → `"mps"`, (2) `torch.cuda.is_available()` → `"cuda"`, (3) varsayılan → `"cpu"`.
2. WHEN cihaz seçimi tamamlandığında, THE Sistem SHALL seçilen cihazın adını (`mps`, `cuda` veya `cpu`) notebook çıktısında `"Kullanılan cihaz: <cihaz_adı>"` formatında göstersin.
3. THE Egitici SHALL her eğitim ve değerlendirme adımında model parametrelerini ve veri batch'lerini `.to(device)` çağrısıyla seçilen cihaza taşısın.
4. IF MPS veya CUDA mevcut değilse, THEN THE Sistem SHALL `"UYARI: GPU bulunamadı, CPU kullanılıyor. Eğitim süresi uzayabilir."` mesajını konsola yazdırsın ve CPU üzerinde çalışmaya devam etsin.

---

### Gereksinim 6: Model Eğitimi

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, her iki modeli de kontrollü ve tekrarlanabilir biçimde eğitmek istiyorum; böylece adil bir karşılaştırma yapabilirim.

#### Kabul Kriterleri

1. THE Egitici SHALL her epoch sonunda eğitim kaybı, doğrulama kaybı, eğitim doğruluğu ve doğrulama doğruluğunu bir Python sözlüğünde (history dict) kaydetsin ve epoch numarasıyla birlikte konsola yazdırsın.
2. THE Egitici SHALL doğrulama kaybı art arda 10 epoch boyunca bir önceki en iyi değerin altına düşmediğinde eğitimi erken durdursun (early stopping, patience=10); erken durdurma tetiklendiğinde hangi epoch'ta durulduğunu konsola yazdırsın.
3. THE Egitici SHALL her epoch sonunda doğrulama kaybı yeni bir en düşük değere ulaştığında model ağırlıklarını `best_<model_adı>.pth` dosyasına kaydetsin; yalnızca en iyi ağırlıklar saklanacak şekilde önceki dosyanın üzerine yazsın.
4. THE Egitici SHALL `torch.optim.lr_scheduler.ReduceLROnPlateau` zamanlayıcısını `mode='min'`, `factor=0.5`, `patience=5` parametreleriyle kullanarak öğrenme oranını doğrulama kaybına göre otomatik azaltsın.
5. THE Egitici SHALL her iki model için maksimum 50 epoch ve aynı erken durdurma kriterini (patience=10) kullanarak eğitimi gerçekleştirsin; CNN_Modeli ve Transfer_Modeli için epoch sayısı farklı olabilir ancak kriter aynı olmalıdır.
6. WHEN eğitim tamamlandığında (erken durdurma veya maksimum epoch nedeniyle), THE Egitici SHALL toplam eğitim süresini dakika ve saniye cinsinden `"Eğitim süresi: X dakika Y saniye"` formatında raporlasın.

---

### Gereksinim 7: Hiperparametre Analizi

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, en az bir hiperparametrenin model performansına etkisini sistematik biçimde incelemek istiyorum; böylece en iyi konfigürasyonu belirleyebilirim.

#### Kabul Kriterleri

1. THE Hiperparametre_Analizci SHALL öğrenme oranı için tam olarak üç değeri (1e-2, 1e-3, 1e-4) CNN_Modeli üzerinde ayrı ayrı eğiterek denesin; her deney bağımsız bir model örneğiyle başlatılsın.
2. THE Hiperparametre_Analizci SHALL her deney için doğrulama weighted-F1 skorunu (sklearn `f1_score(..., average='weighted')`) kaydetsin; F1 hesaplamasında kullanılan eşik değeri 0.5 olsun.
3. THE Hiperparametre_Analizci SHALL deney sonuçlarını öğrenme oranı sütunu ve doğrulama weighted-F1 skoru sütunu içeren bir pandas DataFrame tablosu ve öğrenme oranı (x ekseni, log ölçeği) ile F1 skoru (y ekseni) arasındaki ilişkiyi gösteren bir çizgi grafik olarak görselleştirsin.
4. WHEN tüm deneyler tamamlandığında, THE Hiperparametre_Analizci SHALL en yüksek doğrulama weighted-F1 skorunu veren öğrenme oranını `"En iyi öğrenme oranı: <değer>, Doğrulama F1: <skor>"` formatında konsola yazdırsın.
5. THE Hiperparametre_Analizci SHALL her deney başında `torch.manual_seed(42)` ve `numpy.random.seed(42)` çağrılarını yaparak tekrarlanabilirliği sağlasın.
6. THE Hiperparametre_Analizci SHALL her deney için maksimum 20 epoch ve patience=5 erken durdurma kullanarak makul sürede tamamlansın.

---

### Gereksinim 8: Overfitting Analizi

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, her iki modelin overfitting durumunu görsel ve sayısal olarak analiz etmek istiyorum; böylece modellerin genelleme kapasitesini değerlendirebilirim.

#### Kabul Kriterleri

1. THE Degerlendiricı SHALL her model için eğitim geçmişindeki tüm epoch'lara ait eğitim kaybı ve doğrulama kaybı değerlerini aynı grafik üzerinde farklı renklerle çizsin; x ekseni epoch numarası, y ekseni kayıp değeri olsun.
2. THE Degerlendiricı SHALL her model için eğitim geçmişindeki tüm epoch'lara ait eğitim doğruluğu ve doğrulama doğruluğu değerlerini aynı grafik üzerinde farklı renklerle çizsin; x ekseni epoch numarası, y ekseni doğruluk (%) olsun.
3. THE Degerlendiricı SHALL son 5 epoch'un ortalaması alınarak hesaplanan eğitim kaybı ile doğrulama kaybı arasındaki farkı (overfitting gap) `"Overfitting gap (son 5 epoch ortalaması): <değer>"` formatında raporlasın.
4. THE Degerlendiricı SHALL eğitim doğruluğu ile doğrulama doğruluğu arasındaki farkın %10 puanı aştığı epoch'ları kayıp eğrisi grafiğinde dikey kesikli çizgiyle işaretlesin; böyle bir epoch yoksa `"Overfitting eşiği aşılmadı"` mesajını yazdırsın.
5. THE Degerlendiricı SHALL tartışma bölümünde her model için overfitting gap değerini, erken durdurmanın tetiklenip tetiklenmediğini ve Dropout'un kullanıldığı katmanları sayısal olarak listelesin.

---

### Gereksinim 9: Performans Değerlendirmesi

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, her iki modelin test seti üzerindeki performansını kapsamlı metriklerle ölçmek istiyorum; böylece nesnel bir karşılaştırma yapabilirim.

#### Kabul Kriterleri

1. THE Degerlendiricı SHALL her model için test seti üzerinde Accuracy, Precision (macro ve weighted), Recall (macro ve weighted) ve F1-Score (macro ve weighted) metriklerini `sklearn.metrics` kullanarak hesaplayıp raporlasın; sınıflandırma eşiği 0.5 olsun.
2. THE Degerlendiricı SHALL her model için Karışıklık Matrisini (TP, TN, FP, FN değerleriyle birlikte) `seaborn.heatmap` ile görselleştirsin; x ekseni tahmin edilen etiket, y ekseni gerçek etiket olsun ve hücrelerde sayısal değerler gösterilsin.
3. THE Degerlendiricı SHALL her model için ROC eğrisini ve AUC değerini hesaplayıp her iki modelin eğrilerini aynı grafik üzerinde farklı renklerle çizsin; rastgele sınıflandırıcı referans çizgisi de gösterilsin.
4. THE Degerlendiricı SHALL her model için Precision-Recall eğrisini ve Average Precision (AP) değerini hesaplayıp her iki modelin eğrilerini aynı grafik üzerinde farklı renklerle çizsin.
5. THE Degerlendiricı SHALL CNN_Modeli ve Transfer_Modeli için Accuracy, Macro-F1, Weighted-F1, ROC-AUC ve AP metriklerini içeren yan yana karşılaştırma tablosu oluştursun; tablo pandas DataFrame olarak gösterilsin.
6. WHEN test değerlendirmesi yapılırken, THE Degerlendiricı SHALL `best_<model_adı>.pth` dosyasından en iyi doğrulama kaybına sahip model ağırlıklarını yüklesin; IF bu dosya mevcut değilse THEN `FileNotFoundError` fırlatarak kullanıcıyı bilgilendirsin.

---

### Gereksinim 10: Model Karşılaştırması ve Tartışma

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, iki modelin güçlü ve zayıf yönlerini analiz etmek istiyorum; böylece PCB anomali tespiti için en uygun yaklaşımı belirleyebilirim.

#### Kabul Kriterleri

1. THE Degerlendiricı SHALL her iki model için test setindeki yanlış sınıflandırılan görüntüleri False Positive ve False Negative olarak ayrı ayrı görselleştirsin; her kategoriden en az 5 örnek gösterilsin, görüntülerin üzerinde gerçek etiket ve tahmin edilen etiket yazılsın.
2. THE Degerlendiricı SHALL CNN_Modeli ve Transfer_Modeli için Model Adı, Toplam Parametre Sayısı, Eğitilebilir Parametre Sayısı, Eğitim Süresi (dakika), Test Macro-F1 ve Test ROC-AUC sütunlarını içeren bir karşılaştırma tablosu oluştursun.
3. WHEN tartışma bölümü yazılırken, THE Sistem SHALL hangi modelin daha yüksek Test Macro-F1 skoruna sahip olduğunu belirterek tercih gerekçesini sunacak şekilde; eğer fark ≥ 0.05 ise daha yüksek F1'li modeli, fark < 0.05 ise daha az parametre ve daha kısa eğitim süresine sahip modeli önersın.
4. THE Sistem SHALL tartışma bölümünde CNN_Modeli ve Transfer_Modeli için Precision ve Recall değerlerini karşılaştırarak sınıf dengesizliğinin hangi modeli daha fazla etkilediğini sayısal fark (ΔPrecision, ΔRecall) ile göstersin.

---

### Gereksinim 11: Jupyter Notebook Yapısı ve Raporlama

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, projenin tüm adımlarını tek bir Jupyter Notebook dosyasında adım adım çalıştırılabilir biçimde sunmak istiyorum; böylece proje hem tekrarlanabilir hem de değerlendirilebilir olur.

#### Kabul Kriterleri

1. THE Sistem SHALL tüm kodu ve analizleri tek bir `.ipynb` dosyasında aşağıdaki bölüm sırasıyla sunacak şekilde düzenlesin: (1) Giriş ve Kurulum, (2) Veri Analizi, (3) Model Tasarımı, (4) Deneysel Çalışmalar, (5) Sonuçlar ve Karşılaştırma, (6) Tartışma.
2. THE Sistem SHALL her ana bölümü `## Bölüm N: <Başlık>` formatında bir Markdown hücresiyle başlatsın ve bölümün amacını, kullanılan yöntemi ve beklenen çıktıyı en az 2 cümleyle açıklasın.
3. WHEN notebook bağımlılıkların kurulu olduğu temiz bir ortamda baştan sona sırayla çalıştırıldığında, THE Sistem SHALL herhangi bir `Exception` veya `Error` üretmeden tüm hücreleri başarıyla tamamlasın.
4. THE Sistem SHALL notebook'un ilk çalıştırılabilir hücresinde `torch.manual_seed(42)`, `numpy.random.seed(42)` ve `random.seed(42)` tohumlarını ayarlasın; bu hücre diğer tüm import ve işlem hücrelerinden önce gelsin.
5. THE Sistem SHALL veri seti kök yolunu notebook'un başında `DATA_ROOT = "dataset/pcb1/Data/Images"` şeklinde tek bir değişken olarak tanımlasın; bu değişken dışında sabit kodlanmış yol kullanılmasın.
6. THE Sistem SHALL aynı işlevi gerçekleştiren kod bloğunu birden fazla hücrede kopyalamak yerine yeniden kullanılabilir fonksiyon veya sınıf olarak tanımlasın; aynı mantığın iki veya daha fazla hücrede tekrarlandığı durum bulunmasın.

---

### Gereksinim 12: Sanal Ortam (venv) Kurulumu

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, projeyi izole bir Python sanal ortamında çalıştırmak istiyorum; böylece bağımlılık çakışmaları olmadan tekrarlanabilir bir ortam elde ederim.

#### Kabul Kriterleri

1. THE Sistem SHALL proje kök dizininde `requirements.txt` dosyası içersin; bu dosya `torch>=2.0`, `torchvision>=0.15`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `seaborn>=0.12`, `jupyter>=1.0`, `pandas>=2.0`, `numpy>=1.24`, `Pillow>=9.0`, `ipykernel>=6.0` paketlerini minimum sürümleriyle listelesin.
2. THE Sistem SHALL proje kök dizininde `setup_env.sh` adlı bir kabuk betiği içersin; bu betik sırasıyla `python3 -m venv venv`, `source venv/bin/activate`, `pip install --upgrade pip` ve `pip install -r requirements.txt` komutlarını çalıştırsın.
3. THE Sistem SHALL `setup_env.sh` betiği çalıştırıldıktan sonra `python -m ipykernel install --user --name=venv --display-name "Python (venv)"` komutuyla sanal ortamı Jupyter çekirdeği olarak kaydetsin; böylece notebook `venv` çekirdeğiyle çalıştırılabilsin.
4. IF `venv/` dizini zaten mevcutsa, THEN `setup_env.sh` SHALL mevcut ortamı silmeden yalnızca `pip install -r requirements.txt` adımını yeniden çalıştırsın.
5. THE Sistem SHALL `.gitignore` dosyasında `venv/` dizinini, `__pycache__/`, `*.pth` model ağırlık dosyalarını ve `.ipynb_checkpoints/` dizinini dışlasın.

---

### Gereksinim 13: README Dosyası

**Kullanıcı Hikayesi:** Bir araştırmacı olarak, projeyi ilk kez inceleyen birinin kurulum ve çalıştırma adımlarını kolayca anlayabileceği bir README dosyası istiyorum.

#### Kabul Kriterleri

1. THE Sistem SHALL proje kök dizininde bir `README.md` dosyası içersin.
2. THE README SHALL proje amacını, kullanılan veri setini (SPOT-Diff PCB1) ve kaynak bağlantısını `[SPOT-Diff](https://github.com/amazon-science/spot-diff)` formatında Markdown bağlantısı olarak içersin.
3. THE README SHALL `## Kurulum` bölümünde şu adımları sırasıyla açıklasın: (1) repoyu klonla, (2) `bash setup_env.sh` komutuyla sanal ortamı kur, (3) `source venv/bin/activate` ile ortamı etkinleştir, (4) `jupyter notebook pcb_anomaly_detection.ipynb` ile notebook'u aç ve çekirdeği `Python (venv)` olarak seç.
4. THE README SHALL `dataset/pcb1/Data/Images/Normal/` ve `dataset/pcb1/Data/Images/Anomaly/` dizin yapısını gösteren bir ağaç diyagramı içersin.
5. THE README SHALL her iki modelin test Macro-F1 ve ROC-AUC değerlerini içeren bir Markdown tablosu ile `## Sonuçlar` bölümü içersin; bu bölüm eğitim tamamlandıktan sonra doldurulacak şekilde yer tutucu değerler içerebilir.
