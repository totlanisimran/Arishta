from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from my_codes.abandonedobject import capture_live_scenario, detect_abandonment
from my_codes.CrowdDetection import process_single_image, process_live_capture
from my_codes.ObjectDetection1 import process_image, process_live_capture as process_live_capture_object
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/api/live-capture', methods=['POST'])
def live_capture():
    folder_name = "live_scenarios/temp"
    capture_live_scenario(folder_name)
    results, last_frame_object = detect_abandonment(folder_name)
    
    # Get the latest frame
    latest_frame = cv2.imread(os.path.join(folder_name, "frame_6.jpg"))
    
    # Draw bounding boxes on the last frame
    for obj_name, (x1, y1, x2, y2) in last_frame_object.items():
        color = (0, 0, 255)  # BGR format, red color
        label = obj_name.lower()
        cv2.rectangle(latest_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(latest_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', latest_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Format the message
    object_name = list(results.keys())[0] if results else "No object"
    status = results[object_name].lower() if results else "not detected"
    message = f"The {object_name} is {status}"
    
    return jsonify({
        'results': message,
        'image': image_base64
    })

@app.route('/api/process-images', methods=['POST'])
def process_images():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'})
    
    files = request.files.getlist('images')
    upload_folder = "uploaded_scenarios/temp"
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save uploaded images
    for i, file in enumerate(files):
        file.save(os.path.join(upload_folder, f"frame_{i+1}.jpg"))
    
    results, last_frame_object = detect_abandonment(upload_folder)
    object_name = list(results.keys())[0]
    status = results[object_name].lower()
    message = f"The {object_name} is {status}"
    
    # Get the latest frame with bounding box
    latest_frame = cv2.imread(os.path.join(upload_folder, f"frame_{len(files)}.jpg"))
    # _, buffer = cv2.imencode('.jpg', latest_frame)
    # image_base64 = base64.b64encode(buffer).decode('utf-8')

    for obj_name, (x1, y1, x2, y2) in last_frame_object.items():
        color = (0, 0, 255) 
        label = obj_name.lower()
        cv2.rectangle(latest_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(latest_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buffer = cv2.imencode('.jpg', latest_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'results': message,
        'image': image_base64
    })

@app.route('/api/crowd-detection/upload', methods=['POST'])
def process_uploaded_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    # Read image file into numpy array
    nparr = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'})
    
    results = process_single_image(image)
    
    # Convert processed image to base64
    processed_image_b64 = base64.b64encode(results['processed_image']).decode('utf-8')
    
    return jsonify({
        'people_count': results['people_count'],
        'cluster_count': results['cluster_count'],
        'largest_cluster': results['largest_cluster'],
        'image': processed_image_b64
    })

@app.route('/api/crowd-detection/live', methods=['POST'])
def capture_and_process():
    results, error = process_live_capture()
    
    if error:
        return jsonify({'error': error})
    
    # Convert processed image to base64
    processed_image_b64 = base64.b64encode(results['processed_image']).decode('utf-8')
    
    return jsonify({
        'people_count': results['people_count'],
        'cluster_count': results['cluster_count'],
        'largest_cluster': results['largest_cluster'],
        'image': processed_image_b64
    })

# ...existing code...

@app.route('/api/object-detection/upload', methods=['POST'])
def process_uploaded_image_object():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'})
    
    processed_image, detected_objects = process_image(image)
    _, buffer = cv2.imencode('.jpg', processed_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'detected_objects': detected_objects,
        'image': image_base64
    })

@app.route('/api/object-detection/live', methods=['POST'])
def capture_and_process_object():
    processed_frame, objects = process_live_capture_object()
    
    if isinstance(objects, str):  # Error message
        return jsonify({'error': objects})
    
    _, buffer = cv2.imencode('.jpg', processed_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'detected_objects': objects,
        'image': image_base64
    })

@app.route('/')
def index():
    return "Welcome to the Abandoned Object Detection API!"

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5000)