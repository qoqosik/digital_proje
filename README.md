# Yapay Zeka Tabanlı Yangın ve Duman Tespit Sistemi

Bu proje, görüntülerde **yangın** ve **duman** bölgelerini tespit etmek için **YOLOv8 nesne tespit modeli** kullanan yapay zeka tabanlı bir bilgisayarlı görü sistemidir. Proje; veri seti hazırlama, model eğitimi, değerlendirme metrikleri, çıkarım (inference) ve Streamlit tabanlı web dashboard bileşenlerinden oluşur.

## Projenin Amacı

Bu projenin temel amacı, yapay zeka kullanarak erken yangın ve duman tespiti yapabilen bir sistem geliştirmektir. Sistem, kullanıcı tarafından yüklenen bir görüntüyü analiz eder ve görüntüde yangın veya duman bölgeleri olup olmadığını belirler.

Bu tür bir sistem; güvenlik izleme, kamera tabanlı gözetim sistemleri, orman yangını tespiti ve endüstriyel risk önleme gibi alanlarda kullanılabilir.

## Problem Tanımı

Yangın ve duman tespiti önemli bir bilgisayarlı görü problemidir. Çünkü erken tespit, ciddi hasarların ve tehlikeli durumların önlenmesine yardımcı olabilir. Bu projede, üç sınıfı tespit etmek için YOLO tabanlı bir nesne tespit modeli eğitilmiştir:

- `fire`
- `smoke`
- `other`

`other` sınıfı, modelin tehlikeli olmayan nesneleri ayırt etmesine ve yanlış pozitif tespitleri azaltmasına yardımcı olmak için kullanılmıştır.

## Kullanılan Teknolojiler

- Python
- YOLOv8
- Ultralytics
- OpenCV
- Streamlit
- PyTorch
- Pandas
- Pillow

## Veri Seti

Veri seti Roboflow Universe üzerinden YOLOv8 formatında indirilmiştir.

### Veri Seti Bilgileri

| Bölüm | Görüntü Sayısı |
|---|---:|
| Eğitim (Train) | 4720 |
| Doğrulama (Validation) | 590 |
| Test | 591 |
| **Toplam** | **5901** |

Veri seti eğitim, doğrulama ve test olarak ayrılmıştır. Her görüntü için YOLO formatında karşılık gelen bir etiket dosyası bulunmaktadır.

## Veri Seti Klasör Yapısı

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Model Eğitimi

Model, Ultralytics kütüphanesindeki **YOLOv8n** modeli kullanılarak eğitilmiştir.

### Eğitim Ayarları

| Parametre | Değer |
|---|---|
| Model | YOLOv8n |
| Epoch | 20 |
| Görüntü Boyutu | 640 |
| Batch Size | 8 |
| Sınıflar | fire, smoke, other |

Eğitim komutu:

```bash
python train_model.py
```

## Doğrulama Sonuçları

Eğitim tamamlandıktan sonra model aşağıdaki doğrulama sonuçlarını elde etmiştir:

| Metrik | Değer |
|---|---:|
| Precision | 85.0% |
| Recall | 80.1% |
| mAP@50 | 89.1% |
| mAP@50-95 | 62.8% |

### Sınıf Bazlı Sonuçlar

| Sınıf | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| fire | 87.7% | 84.2% | 92.3% | 67.4% |
| smoke | 83.9% | 80.6% | 88.7% | 59.8% |
| other | 83.6% | 75.3% | 86.3% | 61.2% |

## Web Dashboard

Çıkarım ve demo gösterimi için Streamlit tabanlı bir dashboard geliştirilmiştir.

Dashboard üzerinden kullanıcı şu işlemleri yapabilir:

- Görüntü yükleyebilir
- Confidence threshold değerini ayarlayabilir
- Yangın ve duman tespitini çalıştırabilir
- Orijinal görüntüyü görebilir
- Tespit sonucunu bounding box'lar ile görebilir
- Risk seviyesini görebilir
- Tespit edilen nesne sayılarını görebilir
- Confidence değerlerini tablo halinde inceleyebilir

### Risk Seviyesi Mantığı

| Durum | Risk Seviyesi |
|---|---|
| Yangın tespit edilirse | High Risk |
| Sadece duman tespit edilirse | Warning |
| Yangın veya duman tespit edilmezse | Safe |

## Proje Klasör Yapısı

```text
fire-smoke-detection/
├── app/
│   └── streamlit_app.py
├── models/
│   └── best.pt
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── runs/
├── test_images/
├── split_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

> Not: `dataset/`, `runs/` ve `models/best.pt` gibi büyük dosyalar GitHub deposuna eklenmeyebilir. Bu dosyalar yerel ortamda bulunmalıdır.

## Kurulum

Depoyu klonlayın:

```bash
git clone https://github.com/qoqosik/digital_proje.git
cd digital_proje
```

Sanal ortam oluşturun:

```bash
python -m venv .venv
source .venv/bin/activate
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## Streamlit Uygulamasını Çalıştırma

Uygulamayı çalıştırmak için:

```bash
streamlit run app/streamlit_app.py
```

Daha sonra tarayıcıda aşağıdaki yerel adresi açın:

```text
http://localhost:8501
```

## Çıkarım (Inference) Çalıştırma

Eğitilmiş model ile tahmin yapmak için:

```bash
python -c "from ultralytics import YOLO; model=YOLO('models/best.pt'); model.predict(source='test_images', save=True, conf=0.30)"
```

Tahmin sonuçları aşağıdaki klasöre kaydedilir:

```text
runs/detect/predict/
```

## Modeli Yeniden Eğitme

Modeli yeniden eğitmek için:

```bash
python train_model.py
```

Eğitimden sonra en iyi model dosyası şu konuma kaydedilir:

```text
runs/detect/fire_smoke_demo/weights/best.pt
```

Dashboard üzerinde kullanmak için model dosyası şu konuma kopyalanmalıdır:

```text
models/best.pt
```

Örnek:

```bash
mkdir -p models
cp runs/detect/fire_smoke_demo/weights/best.pt models/best.pt
```

> Not: Eğitim çıktısı farklı bir klasörde oluşursa, `best.pt` dosyasını kendi `runs/detect/.../weights/` klasörünüzden kopyalayın.

## Demo Açıklaması

Demo sürümünde kullanıcı Streamlit arayüzü üzerinden bir görüntü yükler. Eğitilmiş YOLOv8 modeli görüntüyü analiz eder ve yangın, duman veya diğer nesneleri tespit eder. Sistem; bounding box'ları, confidence değerlerini, tespit sayılarını ve genel risk değerlendirmesini ekranda gösterir.

## Notlar

- Veri seti ve eğitilmiş model dosyaları dosya boyutu nedeniyle GitHub deposuna eklenmeyebilir.
- Veri seti Roboflow Universe üzerinden indirilebilir.
- Dashboard'u çalıştırmadan önce eğitilmiş model dosyası `models/best.pt` konumunda bulunmalıdır.
- `other` sınıfı, modelin daha kararlı çalışması ve yanlış pozitif tespitleri azaltması için kullanılmıştır.

## Gelecek Geliştirmeler

Final sürümü için yapılabilecek geliştirmeler:

- Video üzerinde yangın ve duman tespiti ekleme
- Webcam / canlı kamera tespiti ekleme
- Kullanıcı arayüzünü daha da geliştirme
- YOLOv8s gibi daha büyük bir YOLO modeliyle yeniden eğitim yapma
- Tespit geçmişi ekleme
- Tespit raporunu PDF olarak dışa aktarma
- Dashboard'u çevrimiçi olarak yayınlama



## Yazarlar

- **Yerkassyn Zaiymov**
- **Adilet Kairzhanov**
