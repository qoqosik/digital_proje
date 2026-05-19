import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #555;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e9ecef;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    .metric-title {
        font-size: 14px;
        color: #666;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #222;
    }

    .risk-high {
        background-color: #ffe5e5;
        color: #b00020;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #d00000;
        font-size: 22px;
        font-weight: 700;
    }

    .risk-warning {
        background-color: #fff7d6;
        color: #8a5a00;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #f0ad00;
        font-size: 22px;
        font-weight: 700;
    }

    .risk-safe {
        background-color: #e7f8ed;
        color: #146c2e;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #1fa34a;
        font-size: 22px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")

    if not model_path.exists():
        st.error("Model file not found. Please make sure models/best.pt exists.")
        st.stop()

    return YOLO(str(model_path))


model = load_model()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🔥 Project Info")

    st.write(
        """
        This project detects **fire** and **smoke** in images using a trained
        YOLOv8 object detection model.
        """
    )

    st.divider()

    st.subheader("Dataset")
    st.write("Total images: **5901**")
    st.write("Train: **4720**")
    st.write("Validation: **590**")
    st.write("Test: **591**")

    st.divider()

    st.subheader("Model")
    st.write("Architecture: **YOLOv8n**")
    st.write("Epochs: **20**")
    st.write("Classes: **fire, smoke, other**")

    st.divider()

    st.subheader("Validation Metrics")
    st.write("Precision: **85.0%**")
    st.write("Recall: **80.1%**")
    st.write("mAP@50: **89.1%**")
    st.write("mAP@50-95: **62.8%**")

    st.divider()

    st.subheader("How it works")
    st.write(
        """
        1. Upload an image  
        2. Set confidence threshold  
        3. Run detection  
        4. View bounding boxes and risk level  
        """
    )


# -----------------------------
# Main title
# -----------------------------
st.markdown(
    '<div class="main-title">🔥 AI-Based Fire and Smoke Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an image and the trained YOLO model will detect fire and smoke regions.</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Top metric cards
# -----------------------------
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">5901</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">mAP@50</div>
            <div class="metric-value">89.1%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">85.0%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">80.1%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# -----------------------------
# Upload section
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Minimum confidence score required to display a detected object."
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    detect_button = st.button("Detect Fire / Smoke", type="primary")

    if detect_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        with st.spinner("Running YOLO detection..."):
            results = model.predict(
                source=temp_path,
                conf=confidence,
                save=False
            )

        result = results[0]
        annotated_image = result.plot()

        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf_score = float(box.conf[0])

                detections.append(
                    {
                        "Class": class_name,
                        "Confidence": f"{conf_score * 100:.1f}%"
                    }
                )

        fire_count = sum(1 for item in detections if item["Class"] == "fire")
        smoke_count = sum(1 for item in detections if item["Class"] == "smoke")
        other_count = sum(1 for item in detections if item["Class"] == "other")

        # -----------------------------
        # Risk level
        # -----------------------------
        st.subheader("Risk Assessment")

        if fire_count > 0:
            st.markdown(
                '<div class="risk-high">🔥 HIGH RISK — Fire detected</div>',
                unsafe_allow_html=True
            )
        elif smoke_count > 0:
            st.markdown(
                '<div class="risk-warning">💨 WARNING — Smoke detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="risk-safe">✅ SAFE — No fire or smoke detected</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # -----------------------------
        # Images
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Result")
            st.image(annotated_image, channels="BGR", use_container_width=True)

        st.divider()

        # -----------------------------
        # Summary
        # -----------------------------
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Fire detections", fire_count)

        with summary_col2:
            st.metric("Smoke detections", smoke_count)

        with summary_col3:
            st.metric("Other detections", other_count)

        # -----------------------------
        # Detection table
        # -----------------------------
        st.subheader("Detected Objects")

        if len(detections) == 0:
            st.info("No objects detected with the selected confidence threshold.")
        else:
            df = pd.DataFrame(detections)

            # Important classes first
            class_order = {"fire": 0, "smoke": 1, "other": 2}
            df["Sort"] = df["Class"].map(class_order)
            df = df.sort_values(by=["Sort", "Confidence"], ascending=[True, False])
            df = df.drop(columns=["Sort"])
            df = df.reset_index(drop=True)
            st.dataframe(df, use_container_width=True)

        st.divider()

        # -----------------------------
        # Explanation for demo
        # -----------------------------
        st.subheader("Result Explanation")

        if fire_count > 0:
            st.write(
                "The system detected fire regions in the uploaded image. "
                "This indicates a high-risk situation."
            )
        elif smoke_count > 0:
            st.write(
                "The system detected smoke regions in the uploaded image. "
                "This may indicate an early fire risk or hazardous condition."
            )
        else:
            st.write(
                "The system did not detect fire or smoke above the selected confidence threshold."
            )

else:
    st.info("Please upload an image to start detection.")