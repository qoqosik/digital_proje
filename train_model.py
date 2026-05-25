from ultralytics import YOLO
import torch

def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=1000,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name="fire_smoke_final_1000epochs",
        device=device,
    )

if __name__ == "__main__":
    main()