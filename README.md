# 🔥 Yapay Zekâ Tabanlı Yangın ve Duman Tespit Sistemi

## Proje Açıklaması

Bu proje, görüntülerde ve gerçek zamanlı kamera akışında yangın ile duman tespiti yapabilen yapay zekâ destekli bir bilgisayarlı görü sistemi geliştirmek amacıyla hazırlanmıştır.

Sistem, YOLOv8 nesne tespit modeli kullanılarak eğitilmiş ve yangın güvenliği senaryoları için optimize edilmiştir. Kullanıcılar web arayüzü üzerinden görüntü yükleyebilir veya gerçek zamanlı kamera modunu çalıştırarak canlı tespit gerçekleştirebilir.

Proje kapsamında:
- Görüntü tabanlı nesne tespiti
- Gerçek zamanlı kamera analizi
- Risk değerlendirmesi
- Bounding box çizimi
- Güven skoru analizi
özellikleri geliştirilmiştir.

---

# 👨‍💻 Proje Geliştiricileri

- **Yerkassyn Zaiymov**
- **Adilet Kairzhanov**

---

# 🎯 Projenin Amacı

Bu projenin amacı:
- Yangın ve dumanı erken aşamada tespit etmek
- Bilgisayarlı görü tekniklerini gerçek hayatta uygulamak
- Gerçek zamanlı güvenlik sistemleri geliştirmek
- Yapay zekâ tabanlı risk analiz sistemleri oluşturmak

---

# 🧠 Kullanılan Teknolojiler

## Programlama Dili
- Python 3

## Yapay Zekâ ve Bilgisayarlı Görü
- YOLOv8
- Ultralytics
- OpenCV

## Web Arayüzü
- Streamlit

## Veri İşleme
- NumPy
- Pandas
- Pillow

---

# 📂 Veri Seti

Projede yangın ve duman görüntülerinden oluşan özel bir veri seti kullanılmıştır.

## Sınıflar
- fire
- smoke
- other

## Veri Dağılımı
- Toplam görüntü: 5901
- Eğitim verisi: %60
- Doğrulama verisi: %20
- Test verisi: %20

---

# ⚙️ Model Eğitimi

Model eğitimi YOLOv8n mimarisi kullanılarak gerçekleştirilmiştir.

## Eğitim Bilgileri
- Model: YOLOv8n
- Epoch sayısı: 100
- Görüntü boyutu: 640x640
- Framework: PyTorch

---

# 📊 Model Performansı

## Sonuçlar
- Precision: 87.2%
- Recall: 83.5%
- mAP@50: 91.0%
- mAP@50-95: 66.7%

Model başarılı şekilde yangın ve duman bölgelerini tespit edebilmektedir.

---

# 🖼️ Görüntü Tespit Sistemi

Kullanıcı:
1. Bir görüntü yükler
2. Confidence threshold değerini belirler
3. “Detect Fire / Smoke” butonuna basar
4. Sistem nesneleri analiz eder
5. Sonuçlar ekranda gösterilir

Sistem:
- Yangın bölgelerini
- Duman bölgelerini
- Diğer nesneleri
tespit eder ve bounding box ile işaretler.

---

# 🎥 Gerçek Zamanlı Kamera Sistemi

Proje gerçek zamanlı kamera desteğine sahiptir.

Kullanıcı:
- “Launch Real-Time Camera” butonuna basarak sistemi çalıştırabilir.
- Webcam üzerinden canlı yangın ve duman tespiti yapılabilir.

Gerçek zamanlı sistem:
- FPS bilgisi gösterir
- Risk seviyesini analiz eder
- Canlı bounding box çizer
- Anlık nesne tespiti yapar

---

# 🚨 Risk Analizi

Sistem otomatik risk değerlendirmesi yapmaktadır.

## Risk Seviyeleri

### 🔴 HIGH RISK
Yangın tespit edildiğinde gösterilir.

### 🟡 WARNING
Duman tespit edildiğinde gösterilir.

### 🟢 SAFE
Yangın veya duman bulunmadığında gösterilir.

---

# 📁 Proje Yapısı

```bash
digital_proje/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── best.pt
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── realtime_camera.py
├── train_model.py
├── split_dataset.py
├── requirements.txt
└── README.md
```

---

# ▶️ Projeyi Çalıştırma

## 1. Repository Klonlama

```bash
git clone https://github.com/qoqosik/digital_proje.git
cd digital_proje
```

## 2. Gerekli Kütüphaneleri Kurma

```bash
pip install -r requirements.txt
```

## 3. Streamlit Uygulamasını Başlatma

```bash
streamlit run app/streamlit_app.py
```

---

# 📷 Gerçek Zamanlı Kamera Modu

```bash
python realtime_camera.py
```

---

# 💡 Projenin Özellikleri

✅ Yapay zekâ destekli yangın tespiti  
✅ Duman tespiti  
✅ Gerçek zamanlı kamera desteği  
✅ Web tabanlı kullanıcı arayüzü  
✅ YOLOv8 nesne tespiti  
✅ Risk değerlendirme sistemi  
✅ Bounding box görselleştirmesi  
✅ Confidence threshold ayarlama  
✅ Modern Streamlit arayüzü  

---

# 📌 Gelecekte Yapılabilecek Geliştirmeler

- Alarm sistemi entegrasyonu
- Mobil uygulama desteği
- IP kamera entegrasyonu
- Daha büyük veri seti ile eğitim
- Daha yüksek doğruluk oranı
- Cloud tabanlı izleme sistemi
- Çoklu kamera desteği

---

# 📚 Akademik Amaç

Bu proje, bilgisayarlı görü ve yapay zekâ teknolojilerinin gerçek hayat güvenlik problemlerine uygulanmasını göstermek amacıyla geliştirilmiştir.

---

## Yazarlar

- **Yerkassyn Zaiymov**
- **Adilet Kairzhanov**