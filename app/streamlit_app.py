import base64
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


DATASET_SIZE = 5901
TRAIN_IMAGES = 4720
VAL_IMAGES = 590
TEST_IMAGES = 591
MODEL_ARCHITECTURE = "YOLOv8n"
EPOCHS = 100
PRECISION = "87.2%"
RECALL = "83.5%"
MAP_50 = "91.0%"
MAP_50_95 = "66.7%"


st.set_page_config(
    page_title="Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f7fa;
            --panel: #ffffff;
            --panel-soft: #fff7f3;
            --text: #171717;
            --muted: #696f7c;
            --line: #eceff3;
            --fire: #e83923;
            --ember: #ff8a35;
            --safe: #18a957;
            --warning: #d98a00;
            --shadow: 0 12px 28px rgba(20, 24, 31, 0.065);
            --shadow-soft: 0 7px 18px rgba(20, 24, 31, 0.052);
        }

        html, body, [class*="css"] {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .stApp {
            background: var(--bg);
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1080px;
        }

        section[data-testid="stSidebar"] {
            background: #111418;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f7f8fb;
        }

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li {
            color: rgba(247, 248, 251, 0.78);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.12);
        }

        .sidebar-brand {
            padding: 1rem 0 0.35rem;
        }

        .sidebar-brand-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.25;
        }

        .sidebar-brand-subtitle {
            margin-top: 0.35rem;
            color: rgba(255, 255, 255, 0.66);
            font-size: 0.86rem;
        }

        .sidebar-section {
            margin: 1.05rem 0 0.35rem;
            color: #ffffff;
            font-weight: 700;
            letter-spacing: 0;
        }

        .sidebar-kv {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.35rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.9rem;
        }

        .sidebar-kv span:first-child {
            color: rgba(255, 255, 255, 0.68);
        }

        .sidebar-kv span:last-child {
            color: #ffffff;
            font-weight: 700;
            text-align: right;
        }

        .hero {
            position: relative;
            overflow: hidden;
            text-align: center;
            max-width: 1000px;
            margin: 0 auto;
            padding: 1.15rem clamp(1rem, 3vw, 1.6rem);
            border: 1px solid rgba(232, 57, 35, 0.12);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: var(--shadow);
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.65rem;
            border: 1px solid rgba(232, 57, 35, 0.18);
            border-radius: 999px;
            color: #b72918;
            background: #fff7f3;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            margin: 0 auto;
            max-width: 860px;
            color: var(--text);
            font-size: clamp(1.75rem, 4vw, 2.8rem);
            line-height: 1.06;
            font-weight: 800;
            letter-spacing: 0;
        }

        .hero p {
            max-width: 730px;
            margin: 0.5rem auto 0;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.45;
        }

        .metric-card {
            min-height: 108px;
            padding: 0.9rem 0.8rem;
            border-radius: 8px;
            border: 1px solid rgba(232, 57, 35, 0.12);
            background: rgba(255, 255, 255, 0.94);
            box-shadow: var(--shadow-soft);
            text-align: center;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(232, 57, 35, 0.28);
            box-shadow: 0 14px 30px rgba(232, 57, 35, 0.10);
        }

        .metric-icon {
            font-size: 1.35rem;
            line-height: 1;
            margin-bottom: 0.42rem;
        }

        .metric-title {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .metric-value {
            margin-top: 0.25rem;
            color: var(--text);
            font-size: clamp(1.42rem, 2.2vw, 1.85rem);
            font-weight: 800;
            line-height: 1.1;
        }

        .section-title {
            margin: 1.15rem 0 0.42rem;
            color: var(--text);
            font-size: clamp(1.12rem, 2vw, 1.42rem);
            font-weight: 800;
            letter-spacing: 0;
        }

        .section-copy {
            margin: -0.15rem 0 0.6rem;
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.5;
        }

        .explanation-panel {
            padding: 0.85rem 0.95rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: var(--shadow-soft);
            padding: 1rem 1rem 1.05rem;
        }

        .workspace-label {
            margin-bottom: 0.35rem;
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 800;
        }

        .card-title-row {
            margin-bottom: 0.3rem;
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 800;
        }

        .card-copy {
            margin: 0 0 0.75rem;
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.45;
        }

        div.stButton > button {
            min-height: 2.75rem;
            border-radius: 8px;
            border: 0;
            font-weight: 700;
            letter-spacing: 0;
            box-shadow: 0 10px 20px rgba(232, 57, 35, 0.14);
            transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 24px rgba(232, 57, 35, 0.20);
        }

        [data-testid="stFileUploader"] {
            padding: 0.25rem 0.55rem;
            border: 1px dashed rgba(232, 57, 35, 0.28);
            border-radius: 8px;
            background: #fffaf7;
        }

        [data-testid="stFileUploader"] section {
            padding: 0.55rem;
        }

        .alert-card {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.95rem 1.05rem;
            border-radius: 8px;
            box-shadow: var(--shadow-soft);
            border: 1px solid;
            margin: 0.1rem 0 0.75rem;
        }

        .alert-icon {
            font-size: 1.65rem;
            line-height: 1;
        }

        .alert-title {
            font-size: clamp(1.05rem, 2vw, 1.28rem);
            font-weight: 800;
            line-height: 1.2;
        }

        .alert-copy {
            margin-top: 0.2rem;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .alert-high {
            background: #fff1ef;
            border-color: rgba(232, 57, 35, 0.28);
            color: #a82315;
        }

        .alert-warning {
            background: #fff8e8;
            border-color: rgba(217, 138, 0, 0.28);
            color: #8a5700;
        }

        .alert-safe {
            background: #eefaf3;
            border-color: rgba(24, 169, 87, 0.28);
            color: #0f7439;
        }

        .image-card {
            padding: 0.65rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 20px rgba(20, 24, 31, 0.045);
        }

        .image-card-title {
            padding: 0.1rem 0.2rem 0.55rem;
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 800;
        }

        .image-card img {
            display: block;
            width: 100%;
            height: clamp(260px, 32vw, 380px);
            object-fit: contain;
            border-radius: 8px;
            background: #f3f5f8;
        }

        [data-testid="stMetric"] {
            padding: 0.85rem 1rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 6px 16px rgba(20, 24, 31, 0.045);
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 800;
        }

        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }

        .footer {
            margin-top: 1.35rem;
            padding: 1rem;
            text-align: center;
            color: var(--muted);
            border-top: 1px solid var(--line);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .footer strong {
            display: block;
            color: var(--text);
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                text-align: left;
            }

            .hero p {
                margin-left: 0;
            }

            .alert-card {
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")

    if not model_path.exists():
        st.error("Model dosyası bulunamadı. Lütfen models/best.pt dosyasının mevcut olduğundan emin olun.")
        st.stop()

    return YOLO(str(model_path))


def sidebar_key_value(label, value):
    st.markdown(
        f"""
        <div class="sidebar-kv">
            <span>{label}</span>
            <span>{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">🔥 Yapay Zeka Güvenlik Paneli</div>
                <div class="sidebar-brand-subtitle">YOLOv8 yangın ve duman algılama projesi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown('<div class="sidebar-section">Projeye Genel Bakış</div>', unsafe_allow_html=True)
        st.write(
            "Yüklenen görsellerden yangın, duman ve diğer görsel bölgeleri tespit eden hafif bir bilgisayarlı görü paneli."
        )

        st.divider()
        st.markdown('<div class="sidebar-section">Veri Kümesi Bilgisi</div>', unsafe_allow_html=True)
        sidebar_key_value("Toplam görsel", f"{DATASET_SIZE:,}")
        sidebar_key_value("Eğitim", f"{TRAIN_IMAGES:,}")
        sidebar_key_value("Doğrulama", f"{VAL_IMAGES:,}")
        sidebar_key_value("Test", f"{TEST_IMAGES:,}")

        st.divider()
        st.markdown('<div class="sidebar-section">Model Detayları</div>', unsafe_allow_html=True)
        sidebar_key_value("Mimari", MODEL_ARCHITECTURE)
        sidebar_key_value("Epok (Epoch)", str(EPOCHS))
        sidebar_key_value("Sınıflar", "yangın, duman, diğer")

        st.divider()
        st.markdown('<div class="sidebar-section">Doğrulama Metrikleri</div>', unsafe_allow_html=True)
        sidebar_key_value("Keskinlik (Precision)", PRECISION)
        sidebar_key_value("Duyarlılık (Recall)", RECALL)
        sidebar_key_value("mAP@50", MAP_50)
        sidebar_key_value("mAP@50-95", MAP_50_95)

        st.divider()
        st.markdown('<div class="sidebar-section">Nasıl Çalışır?</div>', unsafe_allow_html=True)
        st.write(
            """
            1. Bir görsel yükleyin
            2. Güven eşiğini seçin
            3. YOLOv8 çıkarımını çalıştırın
            4. Risk ve tespitleri inceleyin
            """
        )

        st.divider()
        st.markdown('<div class="sidebar-section">Yazarlar</div>', unsafe_allow_html=True)
        st.write("Yerkassyn Zaiymov  \nAdilet Kairzhanov")


def render_hero():
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">🔥 Bilgisayarlı Görü Güvenlik Sistemi</div>
            <h1>Yapay Zeka Tabanlı Yangın ve Duman Algılama Sistemi</h1>
            <p>YOLOv8 bilgisayarlı görü teknolojisi ile gerçek zamanlı yangın ve duman algılama.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(icon, title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_metrics():
    st.markdown('<div class="section-title">Model Performansı</div>', unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        render_metric_card("📦", "Veri Kümesi Boyutu", f"{DATASET_SIZE:,}")
    with metric_col2:
        render_metric_card("🎯", "mAP@50", MAP_50)
    with metric_col3:
        render_metric_card("✅", "Keskinlik (Precision)", PRECISION)
    with metric_col4:
        render_metric_card("📡", "Duyarlılık (Recall)", RECALL)


def render_realtime_section():
    with st.container(border=True):
        st.markdown('<div class="card-title-row">🎥 Gerçek Zamanlı Algılama</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-copy">
                Canlı yangın ve duman izlemesi için OpenCV web kamerası dedektörünü başlatın.
                Ayrı açılan kamera penceresi, eğitilmiş YOLOv8 modelini yerel olarak kullanır.
            </div>
            """,
            unsafe_allow_html=True,
        )
        button_col, _ = st.columns([0.34, 0.66])
        with button_col:
            if st.button("🎥 Gerçek Zamanlı Kamerayı Başlat", type="secondary", use_container_width=True):
                subprocess.Popen([sys.executable, "realtime_camera.py"])
                st.success("Gerçek zamanlı kamera başarıyla başlatıldı.")


def render_alert(level, icon, title, copy):
    st.markdown(
        f"""
        <div class="alert-card alert-{level}">
            <div class="alert-icon">{icon}</div>
            <div>
                <div class="alert-title">{title}</div>
                <div class="alert-copy">{copy}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_image_card(title, image):
    st.markdown(
        f"""
        <div class="image-card">
            <div class="image-card-title">{title}</div>
            <img src="{image_to_data_uri(image)}" alt="{title}">
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <footer class="footer">
            <strong>Yapay Zeka Tabanlı Yangın ve Duman Algılama Sistemi</strong><br>
            YOLOv8 + Streamlit ile geliştirilmiştir<br>
            Geliştirenler:<br>
            Yerkassyn Zaiymov<br>
            Adilet Kairzhanov
        </footer>
        """,
        unsafe_allow_html=True,
    )


inject_css()
model = load_model()
render_sidebar()
render_hero()
render_top_metrics()

st.markdown('<div class="section-title">Görsel Algılama Çalışma Alanı</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Bir görsel yükleyin ve yapay zeka algılamasını çalıştırın.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="workspace-label">Test Görseli Yükle</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Bir görsel yükleyin",
        type=["jpg", "jpeg", "png", "webp"],
        help="Desteklenen formatlar: JPG, JPEG, PNG ve WEBP.",
    )

    confidence = st.slider(
        "Güven eşiği (Confidence threshold)",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Algılanan bir nesnenin görüntülenmesi için gereken minimum güven skoru.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = None

    _, detect_col, _ = st.columns([0.32, 0.36, 0.32])
    with detect_col:
        detect_button = st.button(
            "🔥 Yangın / Duman Algıla",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

if uploaded_file is not None and detect_button:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    with st.spinner("YOLOv8 algılaması çalıştırılıyor..."):
        results = model.predict(
            source=temp_path,
            conf=confidence,
            save=False,
        )

    result = results[0]
    annotated_image = result.plot()
    annotated_image = Image.fromarray(annotated_image[:, :, ::-1])

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf_score = float(box.conf[0])
            detections.append(
                {
                    "Class": class_name,
                    "Confidence": conf_score,
                }
            )

    fire_count = sum(1 for item in detections if item["Class"] == "fire")
    smoke_count = sum(1 for item in detections if item["Class"] == "smoke")
    other_count = sum(1 for item in detections if item["Class"] == "other")

    st.markdown('<div class="section-title">Sonuçlar</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title-row">Risk Değerlendirmesi</div>', unsafe_allow_html=True)

        if fire_count > 0:
            render_alert(
                "high",
                "🔥",
                "YÜKSEK RİSK",
                "Yüklenen görselde yangın algılandı. Acil müdahale önerilir.",
            )
        elif smoke_count > 0:
            render_alert(
                "warning",
                "💨",
                "UYARI",
                "Duman algılandı. Bu durum erken bir yangın riskine veya tehlikeli bir duruma işaret edebilir.",
            )
        else:
            render_alert(
                "safe",
                "✅",
                "GÜVENLİ",
                "Seçilen güven eşiti üzerinde herhangi bir yangın veya duman algılanmadı.",
            )

        st.markdown('<div class="card-title-row" style="margin-top:0.85rem;">Görsel Karşılaştırma</div>', unsafe_allow_html=True)
        image_col1, image_col2 = st.columns(2)

        with image_col1:
            render_image_card("Orijinal Görsel", image)
        with image_col2:
            render_image_card("YOLOv8 Algılama Sonucu", annotated_image)

        st.markdown('<div class="card-title-row" style="margin-top:0.85rem;">Algılama Özeti</div>', unsafe_allow_html=True)
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Yangın Algılamaları", fire_count)
        with summary_col2:
            st.metric("Duman Algılamaları", smoke_count)
        with summary_col3:
            st.metric("Diğer Algılamalar", other_count)

        st.markdown('<div class="card-title-row" style="margin-top:0.85rem;">Algılanan Nesneler</div>', unsafe_allow_html=True)

        if len(detections) == 0:
            st.info("Seçilen güven eşiğinde hiçbir nesne algılanmadı.")
        else:
            df = pd.DataFrame(detections)
            df = df.sort_values(by="Confidence", ascending=False).reset_index(drop=True)
            df["Confidence"] = df["Confidence"].map(lambda value: f"{value * 100:.1f}%")
            df = df.rename(
                columns={
                    "Class": "Algılanan Nesne",
                    "Confidence": "Güven Skoru",
                }
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="card-title-row" style="margin-top:0.85rem;">Sonuç Açıklaması</div>', unsafe_allow_html=True)
        if fire_count > 0:
            explanation = (
                "Sistem, yüklenen görselde yangın bölgeleri tespit etti. "
                "Bu durum yüksek riskli bir senaryoya işaret eder ve izleme sırasında önceliklendirilmelidir."
            )
        elif smoke_count > 0:
            explanation = (
                "Sistem, yüklenen görselde duman bölgeleri tespit etti. "
                "Bu durum erken bir yangın riskini veya başka bir tehlikeli görsel durumu gösterebilir."
            )
        else:
            explanation = (
                "Sistem, seçilen güven eşiğinin üzerinde yangın veya duman tespit etmedi. "
                "Bu görsel tabanlı analize göre ortam güvenli olarak sınıflandırılmıştır."
            )

        st.markdown(
            f'<div class="explanation-panel">{explanation}</div>',
            unsafe_allow_html=True,
        )
st.markdown('<div class="section-title">🎥 Gerçek Zamanlı Kamera</div>', unsafe_allow_html=True)
render_realtime_section()


render_footer()