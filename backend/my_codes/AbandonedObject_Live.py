import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8l.pt")

# Loop through all 7 images
image_path = "trial_frame7.jpg"
    
    # Read image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load {image_path}")

# Run object detection
results = model(image, conf=0.3)
result = results[0]

# Print detected objects
print(f"\nDetected Objects in {image_path}:")
for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    print(f"{class_name} at ({x1},{y1}) to ({x2},{y2})")

# Optional: Display with bounding boxes
annotated_frame = result.plot()
cv2.imshow(f"{image_path}", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()