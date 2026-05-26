import cv2
import time
from ultralytics import YOLO


# -----------------------------
# Load trained model
# -----------------------------
model = YOLO("models/best.pt")


# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()


print("Real-time fire and smoke detection started.")
print("Press 'q' to quit.")


# -----------------------------
# FPS variables
# -----------------------------
prev_time = 0


# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # -----------------------------
    # Run YOLO prediction
    # -----------------------------
    results = model.predict(
        source=frame,
        conf=0.30,
        save=False,
        verbose=False
    )

    result = results[0]

    # Draw detections
    annotated_frame = result.plot()

    # -----------------------------
    # Detection counters
    # -----------------------------
    fire_count = 0
    smoke_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == "fire":
                fire_count += 1

            elif class_name == "smoke":
                smoke_count += 1

    # -----------------------------
    # Risk level
    # -----------------------------
    if fire_count > 0:
        risk_text = "HIGH RISK"
        risk_color = (0, 0, 255)

    elif smoke_count > 0:
        risk_text = "WARNING"
        risk_color = (0, 255, 255)

    else:
        risk_text = "SAFE"
        risk_color = (0, 255, 0)

    # -----------------------------
    # FPS calculation
    # -----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # -----------------------------
    # Draw overlay text
    # -----------------------------
    cv2.putText(
        annotated_frame,
        f"Risk Level: {risk_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        risk_color,
        3
    )

    cv2.putText(
        annotated_frame,
        f"Fire: {fire_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Smoke: {smoke_count}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Show frame
    # -----------------------------
    cv2.imshow(
        "Real-Time Fire and Smoke Detection",
        annotated_frame
    )

    # -----------------------------
    # Exit on Q
    # -----------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()