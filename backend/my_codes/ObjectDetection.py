import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Path to the single image and its ground truth label
img_path = r"C:\Users\Sanford\Desktop\Arishta\object-detection\yolov8\test.jpg"
gt_path = r"C:\Users\Sanford\Desktop\Arishta\object-detection\yolov8\0000320_01050_d_0000009_jpg.rf.7c3ce7a5df6568f62ac506b2fd72445b.txt"
output_folder = r"C:\Users\Sanford\Desktop\Arishta\object-detection\vivek"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8l.pt')

# Define ONLY the classes you want to detect (COCO class IDs)
ALLOWED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Metrics containers
all_predictions_count = []
all_ground_truths_count = []

# For classification metrics
true_positives = 0
false_positives = 0
false_negatives = 0
total_objects = 0

# For per-class metrics
class_metrics = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in ALLOWED_CLASSES}


def read_ground_truth(file_path):
    boxes = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.split())
            boxes.append((int(class_id), x_center, y_center, width, height))
    return boxes


if os.path.exists(gt_path):
    # Run YOLO inference
    results = model(img_path, conf=0.1, iou=0.5, classes=list(ALLOWED_CLASSES.keys()))
    img = cv2.imread(img_path)
    predictions = []
    pred_classes = []
    gt_classes = []

    # Read ground truth boxes
    ground_truths = read_ground_truth(gt_path)
    # Filter ground truths to only include allowed classes
    filtered_ground_truths = [gt for gt in ground_truths if gt[0] in ALLOWED_CLASSES]

    # Track which ground truths have been matched
    matched_gt = [False] * len(filtered_ground_truths)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())

            # ONLY process if class is in our allowed list
            if cls_id in ALLOWED_CLASSES:
                label = ALLOWED_CLASSES[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                predictions.append((cls_id, x1, y1, x2, y2))
                pred_classes.append(cls_id)

                # Draw bounding box
                color = (0, 255, 0)  # Green for predictions
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Match predictions to ground truths (class match only for simplicity)
                matched = False
                for i, gt in enumerate(filtered_ground_truths):
                    if not matched_gt[i] and gt[0] == cls_id:
                        matched_gt[i] = True
                        matched = True
                        break

                if matched:
                    true_positives += 1
                    class_metrics[cls_id]["tp"] += 1
                else:
                    false_positives += 1
                    class_metrics[cls_id]["fp"] += 1

    # Collect ground truth classes
    for gt in filtered_ground_truths:
        gt_classes.append(gt[0])

    # Count false negatives
    for i, matched in enumerate(matched_gt):
        if not matched:
            false_negatives += 1
            class_metrics[filtered_ground_truths[i][0]]["fn"] += 1

    all_predictions_count.append(len(predictions))
    all_ground_truths_count.append(len(filtered_ground_truths))
    total_objects += len(filtered_ground_truths)

    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, img)

    # === METRICS ===
    if all_predictions_count and all_ground_truths_count:
        mse = mean_squared_error(all_ground_truths_count, all_predictions_count)
        rmse = sqrt(mse)
        mae = mean_absolute_error(all_ground_truths_count, all_predictions_count)

        relative_errors = []
        for gt, pred in zip(all_ground_truths_count, all_predictions_count):
            if gt != 0:
                relative_errors.append(abs(gt - pred) / gt)
        are = sum(relative_errors) / len(relative_errors) if relative_errors else 0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / total_objects if total_objects > 0 else 0

        print("=== Object Count Evaluation Metrics ===")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Average Relative Error (ARE): {are:.4f} or {are * 100:.2f}%")

        print("\n=== Detection Quality Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        print("\n=== Per-Class Metrics ===")
        for class_id, metrics in class_metrics.items():
            class_name = ALLOWED_CLASSES[class_id]
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]

            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (
                    class_precision + class_recall) > 0 else 0

            print(f"\nClass: {class_name} (ID: {class_id})")
            print(f"  True Positives: {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  Precision: {class_precision:.4f}")
            print(f"  Recall: {class_recall:.4f}")
            print(f"  F1 Score: {class_f1:.4f}")
    else:
        print("No valid detections found for the specified classes.")
else:
    print("Ground truth label file not found for this image.")