import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8l.pt')

# Define classes you want to detect
ALLOWED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def process_image(image):
    """Process a single image and return results"""
    results = model(image, conf=0.25, classes=list(ALLOWED_CLASSES.keys()))
    
    detected_objects = []
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in ALLOWED_CLASSES:
                label = ALLOWED_CLASSES[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                
                # Draw bounding box
                color = (0, 255, 0)  # Green for predictions
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detected_objects.append({
                    "class": label,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2]
                })
    
    return image, detected_objects

def process_live_capture():
    """Process live camera feed"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Failed to open camera"
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, "Failed to capture image"
    
    processed_frame, objects = process_image(frame)
    return processed_frame, objects