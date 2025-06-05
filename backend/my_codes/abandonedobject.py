import cv2
import time
import os
from ultralytics import YOLO
import cv2
import os

harmful_categories = ['car', 'motorcycle', 'truck', 'bus', 'backpack', 
                      'handbag', 'suitcase', 'sports ball', 'knife', 'scissors']

ground_truth = {
    'scenario1': {'bag': 'ABANDONED', 'dining table': 'NOT ABANDONED'},
    'scenario2': {'bag': 'ABANDONED'},
    'scenario3': {'bag': 'ABANDONED', 'chair': 'NOT ABANDONED'},
    'scenario4': {'bag': 'ABANDONED'},
    'scenario5': {'bag': 'ABANDONED'},
    'scenario6': {'box': 'ABANDONED'},
    'live1': {'bag':'ABANDONED'},
    'live2': {'scissors':'ABANDONED'},
    'live3': {'bag':'ABANDONED'},
    'live4': {'bag':'NOT ABANDONED'},
    'live5': {'bag':'NOT ABANDONED'},
    'live6': {'bag':'NOT ABANDONED'},
    'test_scenario': {'bag': 'ABANDONED'}
}

model = YOLO("yolov8l.pt")

def capture_live_scenario(folder_name, num_frames=6, delay=1):
    os.makedirs(folder_name, exist_ok=True)
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    for i in range(1, num_frames + 1):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(folder_name, f"frame_{i}.jpg"), frame)
            print(f"Frame {i} captured")
        time.sleep(delay)
    cap.release()

# capture_live_scenario("live_scenarios/live6")

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    iou = inter_area / union_area
    return iou

def get_harmful_objects(image):
    results = model(image, conf=0.2)
    result = results[0]
    detected_objects = {}
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if class_name in harmful_categories:
            class_name = 'bag' if class_name in ['handbag','suitcase','backpack'] else class_name
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_objects[class_name] = (x1, y1, x2, y2)
    return detected_objects

def detect_abandonment(folder_path):
    # files = sorted(os.listdir(folder_path))  # Ensure sorted frames
    files = os.listdir(folder_path)
    previous_objects = get_harmful_objects(os.path.join(folder_path, files[0]))
    stable_counts = {obj: 1 for obj in previous_objects}

    for file in files[1:]:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        current_objects = get_harmful_objects(image)
        for obj in current_objects:
            if obj not in stable_counts:
                stable_counts[obj]=1

        for obj, prev_box in previous_objects.items():
            if obj in current_objects:
                iou = compute_iou(prev_box, current_objects[obj])
                if iou > 0.8:
                    stable_counts[obj]+=1
                else:
                    stable_counts[obj]=1 #Reset to 1 as object was moved
            else: #if not in frame - it is moved (edge case 1)
                stable_counts.pop(obj,None)
        previous_objects = current_objects.copy()
    
    # Convert to final label
    print(f"No of frames obj is present in: {stable_counts}")
    return {obj: ('NOT ABANDONED' if count<6 else 'ABANDONED') for obj, count in stable_counts.items()}, current_objects

# all_scenarios = [f"scenarios/scenario{i}" for i in range(1, 7)] + [f"live_scenarios/live{i}" for i in range(1, 7)]


# predicted_labels = {}

# for scenario_path in all_scenarios:
#     folder_name = os.path.basename(scenario_path)
#     predicted_labels[folder_name] = detect_abandonment(scenario_path)

# correct = 0
# total = 0

# for scenario in predicted_labels:
#     total+=1 #assume only one abandoned object per scenario
#     preds = predicted_labels[scenario]
#     truths = ground_truth.get(scenario, {})
#     print(f"Prediction:{preds}")
#     print(f"Ground truth:{truths}")

#     for obj in preds:
#         if obj in truths:
#             if preds[obj] == truths[obj]:
#                 correct += 1

# accuracy = correct / total if total > 0 else 0
# print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")