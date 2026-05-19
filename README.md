# AI-Based Fire and Smoke Detection System

This project is an AI-based computer vision system that detects **fire** and **smoke** regions in images using a **YOLOv8 object detection model**. The system includes dataset preparation, model training, evaluation metrics, inference, and a Streamlit-based web dashboard.

## Project Purpose

The main purpose of this project is to develop an early fire and smoke detection system using artificial intelligence. The system analyzes an uploaded image and identifies whether there are fire or smoke regions in the scene.

This type of system can be useful for safety monitoring, surveillance systems, forest fire detection, and industrial risk prevention.

## Problem Definition

Fire and smoke detection is an important computer vision problem because early detection can help prevent serious damage and danger. In this project, a YOLO-based object detection model was trained to detect three classes:

- `fire`
- `smoke`
- `other`

The `other` class is used to help the model distinguish non-dangerous objects and reduce false positive detections.

## Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV
- Streamlit
- PyTorch
- Pandas
- Pillow

## Dataset

The dataset was downloaded from Roboflow Universe in YOLOv8 format.

### Dataset Information

| Split | Number of Images |
|---|---:|
| Train | 4720 |
| Validation | 590 |
| Test | 591 |
| **Total** | **5901** |

The dataset was divided into training, validation, and test sets. Each image has a corresponding YOLO annotation file.

## Dataset Structure

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

## Model Training

The model was trained using **YOLOv8n** from the Ultralytics library.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Epochs | 20 |
| Image Size | 640 |
| Batch Size | 8 |
| Classes | fire, smoke, other |

Training command:

```bash
python train_model.py
```

## Validation Results

After training, the model achieved the following validation results:

| Metric | Value |
|---|---:|
| Precision | 85.0% |
| Recall | 80.1% |
| mAP@50 | 89.1% |
| mAP@50-95 | 62.8% |

### Class-Based Results

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| fire | 87.7% | 84.2% | 92.3% | 67.4% |
| smoke | 83.9% | 80.6% | 88.7% | 59.8% |
| other | 83.6% | 75.3% | 86.3% | 61.2% |

## Web Dashboard

A Streamlit dashboard was developed for inference and demonstration.

The dashboard allows the user to:

- Upload an image
- Set a confidence threshold
- Run fire and smoke detection
- View the original image
- View the detection result with bounding boxes
- See the risk level
- See detected object counts
- View confidence scores in a table

### Risk Level Logic

| Condition | Risk Level |
|---|---|
| Fire detected | High Risk |
| Smoke detected only | Warning |
| No fire or smoke detected | Safe |

## Project Structure

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
├── demo_images/
├── split_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/qoqosik/digital_proje.git
cd digital_proje
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

Run the application:

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL in your browser:

```text
http://localhost:8501
```

## Running Inference

The trained model can be used for prediction with:

```bash
python -c "from ultralytics import YOLO; model=YOLO('models/best.pt'); model.predict(source='demo_images', save=True, conf=0.30)"
```

Prediction results will be saved under:

```text
runs/detect/predict/
```

## Training the Model

To train the model again:

```bash
python train_model.py
```

After training, the best model is saved as:

```text
runs/detect/fire_smoke_demo/weights/best.pt
```

For the dashboard, the model file should be copied to:

```text
models/best.pt
```

Example:

```bash
mkdir -p models
cp runs/detect/fire_smoke_demo/weights/best.pt models/best.pt
```

> Note: If your training output folder is different, copy `best.pt` from your actual `runs/detect/.../weights/` directory.

## Demo Explanation

In the demo version, the user uploads an image through the Streamlit interface. The trained YOLOv8 model analyzes the image and detects fire, smoke, or other objects. The system displays bounding boxes, confidence scores, detection counts, and an overall risk assessment.

## Notes

- The dataset and trained model files may not be included in the GitHub repository because of file size limitations.
- The dataset can be downloaded from Roboflow Universe.
- The trained model file should be placed in `models/best.pt` before running the dashboard.
- The `other` class is included to improve detection stability and reduce false positives.

## Future Improvements

Possible improvements for the final version:

- Add video detection support
- Add webcam/live camera detection
- Improve the user interface design
- Train a larger YOLO model such as YOLOv8s
- Add detection history
- Export detection reports as PDF
- Deploy the dashboard online

## Authors

- **Yerkassyn Zaiymov**
- **Adilet Kairzhanov**
