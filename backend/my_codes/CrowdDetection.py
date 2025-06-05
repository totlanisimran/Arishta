# final final
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN

# Global variables for clustering parameters
EPS = 300  # Distance threshold for DBSCAN clustering
MIN_CROWD_SIZE = 1  # Minimum number of people to define a crowd

def process_shanghaitech_dataset(image_dir, gt_dir, output_dir="results", max_people=50):
    """Process only images with < max_people ground truth counts."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize YOLO model
    model = YOLO('yolov8l.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Collect image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    results = []
    skipped_count = 0

    for img_file in image_files:
        print(f"\nProcessing {img_file}...")
        img_path = os.path.join(image_dir, img_file)
        gt_file = 'GT_' + os.path.splitext(img_file)[0] + '.mat'
        gt_path = os.path.join(gt_dir, gt_file)

        try:
            # Get ground truth count FIRST
            gt_count = get_shanghaitech_gt_count(gt_path)

            # Skip if too crowded
            if gt_count > max_people:
                print(f"Skipping {img_file} (GT count {gt_count} > {max_people})")
                skipped_count += 1
                continue

            # Only process if under threshold
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not read image {img_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Rest of your processing...
            people_boxes, clusters = detect_people_and_clusters(model, image_rgb)
            detected_people = len(people_boxes)
            absolute_error = abs(detected_people - gt_count)
            relative_error = absolute_error / max(gt_count, 1)

            results.append({
                'image': img_file,
                'ground_truth': gt_count,
                'detected': detected_people,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'clusters': len(clusters),
                'largest_cluster': max([c['size'] for c in clusters], default=0)
            })

            # Pass both people_boxes and clusters to visualization
            visualize_results(image_rgb, people_boxes, clusters,
                            os.path.join(output_dir, f"result_{img_file}"))
            print(f"Ground Truth: {gt_count}, Detected: {detected_people}, Error: {absolute_error}")

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue

    print(f"\nSkipped {skipped_count} images with >{max_people} people")
    return analyze_results(results, output_dir)

def get_shanghaitech_gt_count(gt_path):
    """Extracts count from your specific MAT file structure."""
    try:
        mat = loadmat(gt_path)

        # Extract the nested structure
        info = mat['image_info'][0,0]

        # The count is in the 'number' field of the first element
        count_array = info[0][0][1]  # Gets the [[723]] array
        return int(count_array[0,0])  # Extracts 723 from [[723]]

    except Exception as e:
        print(f"Error loading {os.path.basename(gt_path)}: {str(e)}")
        return 0

def detect_people_and_clusters(model, image):
    """Detect people and cluster them with improved debugging."""
    # Perform detection
    results = model(image)

    # Extract person detections
    people_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls.item()) == 0 and box.conf.item() > 0.1:  # Class 0 is person
                x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                people_boxes.append({'box': (x1, y1, x2, y2), 'position': center})

    if not people_boxes:
        return people_boxes, []

    # Cluster using DBSCAN for better density-based clustering
    positions = np.array([p['position'] for p in people_boxes])

    # DEBUG: Print some information about the positions
    print(f"Number of people detected: {len(positions)}")
    if len(positions) > 1:
        min_dist = float('inf')
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)
        print(f"Minimum distance between any two people: {min_dist:.2f} pixels")
        print(f"Current eps value: {EPS}")

    # Use global EPS value and set min_samples to 2 which is the minimum needed for clustering
    db = DBSCAN(eps=EPS, min_samples=2).fit(positions)
    labels = db.labels_

    # DEBUG: Print information about the clustering result
    print(f"Number of unique labels: {len(set(labels))}")
    print(f"Number of noise points (label -1): {np.sum(labels == -1)}")

    # Process clusters
    clusters = []
    for cluster_id in set(labels):
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_size = len(cluster_indices)

        # Apply MIN_CROWD_SIZE filter here instead of in DBSCAN
        if cluster_size < MIN_CROWD_SIZE:
            print(f"Cluster size {cluster_size} < min_crowd_size {MIN_CROWD_SIZE}, skipping")
            continue

        cluster_positions = positions[cluster_indices]
        centroid = np.mean(cluster_positions, axis=0).astype(int)
        max_distance = max(np.linalg.norm(pos - centroid) for pos in cluster_positions)

        clusters.append({
            'id': cluster_id,
            'size': cluster_size,
            'centroid': tuple(centroid),
            'radius': max_distance,
            'is_crowd': cluster_size >= MIN_CROWD_SIZE,
            'members': cluster_indices.tolist()
        })

    print(f"Number of clusters formed: {len(clusters)}")
    return people_boxes, clusters

def visualize_with_distance_connections(image, people_boxes, clusters, output_path):
    """Enhanced visualization that shows distances between nearby people."""
    plt.figure(figsize=(16, 12))
    plt.imshow(image)

    positions = np.array([p['position'] for p in people_boxes])

    # Draw connections between people who are within EPS distance
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            p1 = positions[i]
            p2 = positions[j]
            dist = np.linalg.norm(p1 - p2)
            if dist <= EPS:
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.3, linewidth=1)
                # Optionally label the connections with distances
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                plt.text(mid_x, mid_y, f"{dist:.0f}", fontsize=8, color='blue')

    # Create a colormap for clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(clusters), 1)))

    # First draw all individual detections with light gray boxes
    for person in people_boxes:
        x1, y1, x2, y2 = person['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1, edgecolor='lightgray', facecolor='none')
        plt.gca().add_patch(rect)

    # Draw noise points (not in any cluster)
    all_cluster_members = set()
    for cluster in clusters:
        all_cluster_members.update(cluster['members'])

    noise_indices = set(range(len(people_boxes))) - all_cluster_members
    for idx in noise_indices:
        x1, y1, x2, y2 = people_boxes[idx]['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1.5, edgecolor='gray', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x1, y1-5, "Single", color='black', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))

    # Then draw each cluster with distinct colors
    for i, cluster in enumerate(clusters):
        color = colors[i]

        # Draw bounding boxes for members of this cluster
        for idx in cluster['members']:
            x1, y1, x2, y2 = people_boxes[idx]['box']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.5, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)

        # Draw cluster boundary as a circle
        centroid = cluster['centroid']
        radius = cluster['radius']
        circle = plt.Circle(centroid, radius, color=color,
                           fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)

        # Add label with cluster info
        crowd_text = " (CROWD)" if cluster['is_crowd'] else ""
        plt.text(centroid[0], centroid[1]-10,
                f"Cluster {cluster['id']+1}: {cluster['size']} people{crowd_text}",
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor=color, alpha=0.7))

    plt.text(50, 50, f"DBSCAN Parameters: eps={EPS}, min_points_in_cluster={MIN_CROWD_SIZE}",
            fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

    plt.title(f"Crowd Detection - {len(people_boxes)} people detected, {len(clusters)} clusters found")
    plt.axis('off')
    plt.show()
    # plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def visualize_results(image, people_boxes, clusters, output_path):
    """Visualize detection results with clusters."""
    plt.figure(figsize=(16, 12))
    plt.imshow(image)

    # Create a colormap for clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(clusters), 1)))

    # First draw all individual detections with light gray boxes
    for person in people_boxes:
        x1, y1, x2, y2 = person['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1, edgecolor='lightgray', facecolor='none')
        plt.gca().add_patch(rect)

    # Draw noise points (not in any cluster)
    all_cluster_members = set()
    for cluster in clusters:
        all_cluster_members.update(cluster['members'])

    noise_indices = set(range(len(people_boxes))) - all_cluster_members
    for idx in noise_indices:
        x1, y1, x2, y2 = people_boxes[idx]['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1.5, edgecolor='gray', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x1, y1-5, "Single", color='black', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))

    # Then draw each cluster with distinct colors
    for i, cluster in enumerate(clusters):
        color = colors[i]

        # Draw bounding boxes for members of this cluster
        for idx in cluster['members']:
            x1, y1, x2, y2 = people_boxes[idx]['box']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.5, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)

        # Draw cluster boundary as a circle
        centroid = cluster['centroid']
        radius = cluster['radius']
        circle = plt.Circle(centroid, radius, color=color,
                           fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)

        # Add label with cluster info
        crowd_text = " (CROWD)" if cluster['is_crowd'] else ""
        plt.text(centroid[0], centroid[1]-10,
                f"Cluster {cluster['id']+1}: {cluster['size']} people{crowd_text}",
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor=color, alpha=0.7))

    # Add additional visualization to show eps radius
    if len(people_boxes) > 0:
        plt.text(50, 50, f"DBSCAN Parameters: eps={EPS}, min_samples={MIN_CROWD_SIZE}",
                fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

    plt.title(f"Crowd Detection - {len(people_boxes)} people detected, {len(clusters)} clusters found")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def analyze_results(results, output_dir, count_threshold=0.6):
    """Analyze and save results with additional metrics like F1 score and accuracy."""
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid results were generated.")
        return results_df

    # Calculate basic metrics
    mse = mean_squared_error(results_df['ground_truth'], results_df['detected'])
    rmse = sqrt(mse)
    mae = results_df['absolute_error'].mean()
    avg_rel_error = results_df['relative_error'].mean() * 100

    # Calculate F1 score and accuracy
    # A prediction is considered "correct" if relative error is below threshold
    results_df['is_correct'] = results_df['relative_error'] <= count_threshold
    accuracy = results_df['is_correct'].mean() * 100

    # For F1 score, define precision and recall
    # Precision: How many of the detected people were actually there
    # Recall: How many of the actual people were detected

    # Calculate total counts
    total_detected = results_df['detected'].sum()
    total_gt = results_df['ground_truth'].sum()

    # True positives: Min of detected and ground truth for each image
    results_df['true_positive'] = results_df.apply(
        lambda row: min(row['detected'], row['ground_truth']), axis=1)
    total_tp = results_df['true_positive'].sum()

    # Calculate precision, recall, F1
    precision = total_tp / total_detected if total_detected > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n=== Evaluation Metrics ===")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Average Relative Error: {avg_rel_error:.2f}%")
    print(f"Accuracy (within {count_threshold*100}% error): {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Save results
    results_csv = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

    # Create and save summary visualizations
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['ground_truth'], results_df['detected'],
                c=results_df['is_correct'].map({True: 'green', False: 'red'}))
    plt.plot([0, results_df['ground_truth'].max()], [0, results_df['ground_truth'].max()], 'b--')

    # Add threshold lines
    max_count = max(results_df['ground_truth'].max(), results_df['detected'].max())
    x = np.linspace(0, max_count, 100)
    plt.plot(x, x*(1+count_threshold), 'g--', alpha=0.5)
    plt.plot(x, x*(1-count_threshold), 'g--', alpha=0.5)

    plt.xlabel('Ground Truth Count')
    plt.ylabel('Detected Count')
    plt.title('Detection Performance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "detection_performance.png"))

    # Add confusion matrix-like visualization
    plt.figure(figsize=(8, 6))
    plt.bar(['Precision', 'Recall', 'F1 Score', 'Accuracy(%)'],
            [precision, recall, f1_score, accuracy/100], color=['blue', 'green', 'purple', 'orange'])
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
    plt.close()

    return results_df

def test_single_image(image_path, gt_path=None):
    """Process a single image for testing purposes"""
    # Initialize YOLO model
    model = YOLO('yolov8x.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get ground truth if available
    gt_count = 0
    if gt_path and os.path.exists(gt_path):
        gt_count = get_shanghaitech_gt_count(gt_path)

    # Detect people and clusters
    people_boxes, clusters = detect_people_and_clusters(model, image_rgb)

    # Visualize with debug info
    os.makedirs("debug_output", exist_ok=True)
    output_path = os.path.join("debug_output", f"debug_eps{EPS}_min{MIN_CROWD_SIZE}.jpg")
    visualize_with_distance_connections(image_rgb, people_boxes, clusters, output_path)

    print(f"Detected {len(people_boxes)} people in {len(clusters)} clusters")
    print(f"Ground truth: {gt_count}")
    print(f"Results saved to {output_path}")
    return people_boxes, clusters


def test_single_image_live(eps=200, min_crowd_size=3, camera_id=0):
    """Perform crowd detection on live camera feed"""
    print("Starting live camera demo... Press 'q' to quit")

    # Initialize YOLO model
    model = YOLO('yolov8l.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Initialize camera
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    # If the default camera doesn't work, try other camera IDs
    if not cap.isOpened():
        print(f"Could not open camera with ID {camera_id}, trying alternative IDs...")
        for i in range(1, 5):  # Try camera IDs 1-4
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_id = i
                print(f"Successfully opened camera with ID {camera_id}")
                break

    if not cap.isOpened():
        print("Error: Could not open any camera. Please check your camera connection and permissions.")
        return

    os.makedirs("debug_output", exist_ok=True)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Convert to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect people and clusters
        people_boxes, clusters = detect_people_and_clusters(model, frame_rgb, eps, min_crowd_size)

        # Create visualization on a copy of the frame
        plt.figure(figsize=(12, 8))
        plt.imshow(frame_rgb)

        # Create a colormap for clusters
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(clusters), 1)))

        # Draw individual detections and clusters
        for person in people_boxes:
            x1, y1, x2, y2 = person['box']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1, edgecolor='lightgray', facecolor='none')
            plt.gca().add_patch(rect)

        # Draw clusters
        all_cluster_members = set()
        for cluster in clusters:
            all_cluster_members.update(cluster['members'])

        # Draw noise points
        noise_indices = set(range(len(people_boxes))) - all_cluster_members
        for idx in noise_indices:
            x1, y1, x2, y2 = people_boxes[idx]['box']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.5, edgecolor='gray', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, "Single", color='black', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))

        # Draw each cluster with distinct colors
        for i, cluster in enumerate(clusters):
            color = colors[i]

            # Draw bounding boxes for members of this cluster
            for idx in cluster['members']:
                x1, y1, x2, y2 = people_boxes[idx]['box']
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=1.5, edgecolor=color, facecolor='none')
                plt.gca().add_patch(rect)

            # Draw cluster boundary as a circle
            centroid = cluster['centroid']
            radius = cluster['radius']
            circle = plt.Circle(centroid, radius, color=color,
                               fill=False, linestyle='--', linewidth=2)
            plt.gca().add_patch(circle)

            # Add label with cluster info
            crowd_text = " (CROWD)" if cluster['is_crowd'] else ""
            plt.text(centroid[0], centroid[1]-10,
                    f"Cluster {cluster['id']+1}: {cluster['size']} people{crowd_text}",
                    color='white', fontsize=10, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.7))

        plt.text(50, 50, f"DBSCAN Parameters: eps={eps}, min_points_in_cluster={min_crowd_size}",
                fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

        plt.title(f"Live Crowd Detection - {len(people_boxes)} people detected, {len(clusters)} clusters found")
        plt.axis('off')

        # Save the visualization
        output_path = os.path.join("debug_output", "live_detection.jpg")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Display the saved image
        display_image = cv2.imread(output_path)
        cv2.imshow('Live Crowd Detection (Press q to quit)', cv2.resize(display_image, (1280, 720)))

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Live demo ended")

# def test_single_image_live1(eps=200, min_crowd_size=1):
#     """Perform crowd detection on live camera feed in Colab with proper cluster visualization"""
#     from IPython.display import display, Javascript, Image, clear_output
#     from google.colab.output import eval_js
#     from base64 import b64decode
#     import time
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches

#     print("Starting live camera demo in Colab...")

#     # Initialize YOLO model
#     model = YOLO('yolov8l.pt')
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)

#     os.makedirs("debug_output", exist_ok=True)

#     def take_photo(quality=0.8):
#         js = Javascript('''
#             async function takePhoto(quality) {
#                 const div = document.createElement('div');
#                 const capture = document.createElement('button');
#                 capture.textContent = 'Capture';
#                 div.appendChild(capture);

#                 const video = document.createElement('video');
#                 video.style.display = 'block';
#                 const stream = await navigator.mediaDevices.getUserMedia({video: true});

#                 document.body.appendChild(div);
#                 div.appendChild(video);
#                 video.srcObject = stream;
#                 await video.play();

#                 google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

#                 await new Promise((resolve) => capture.onclick = resolve);

#                 const canvas = document.createElement('canvas');
#                 canvas.width = video.videoWidth;
#                 canvas.height = video.videoHeight;
#                 canvas.getContext('2d').drawImage(video, 0, 0);
#                 stream.getVideoTracks()[0].stop();
#                 div.remove();
#                 return canvas.toDataURL('image/jpeg', quality);
#             }
#             ''')
#         display(js)
#         data = eval_js(f'takePhoto({quality})')
#         binary = b64decode(data.split(',')[1])
#         filename = 'temp_capture.jpg'
#         with open(filename, 'wb') as f:
#             f.write(binary)
#         return filename

#     try:
#         while True:
#             print("\nPreparing to capture image...")
#             input("Press Enter to capture an image (or Ctrl+C to stop)...")

#             # Clear previous outputs
#             clear_output(wait=True)

#             # Capture image
#             filename = take_photo()
#             print(f"Image captured and saved to {filename}")

#             # Display the captured image
#             display(Image(filename))

#             # Process the image
#             image = cv2.imread(filename)
#             if image is None:
#                 raise ValueError("Could not read the captured image")
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Detect people and clusters
#             people_boxes, clusters = detect_people_and_clusters(model, image_rgb)

#             # Create figure for visualization
#             plt.figure(figsize=(5, 3))
#             plt.imshow(image_rgb)
#             ax = plt.gca()

#             # Draw individual detections with light gray boxes
#             for person in people_boxes:
#                 x1, y1, x2, y2 = person['box']
#                 rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
#                                        linewidth=1, edgecolor='lightgray', facecolor='none')
#                 ax.add_patch(rect)

#             # Create a colormap for clusters
#             colors = plt.cm.rainbow(np.linspace(0, 1, max(len(clusters), 1)))


#             # Special case: Show circles around every person when min_crowd_size=1
#             if min_crowd_size <= 1 and len(clusters) == 0 and len(people_boxes) > 0:
#                 clusters = [{
#                     'id': i,
#                     'size': 1,
#                     'centroid': person['position'],
#                     'radius': max((x2-x1)//2, (y2-y1)//2, 20),  # Minimum radius of 20px
#                     'is_crowd': False,
#                     'members': [i]
#                 } for i, person in enumerate(people_boxes)]
#                 colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

#           # Draw each cluster with distinct colors
#             for i, cluster in enumerate(clusters):
#                 color = colors[i]

#             # Draw clusters if any exist
#             if clusters:
#                 all_cluster_members = set()
#                 for cluster in clusters:
#                     all_cluster_members.update(cluster['members'])

#                 # Draw noise points (not in any cluster)
#                 noise_indices = set(range(len(people_boxes))) - all_cluster_members
#                 for idx in noise_indices:
#                     x1, y1, x2, y2 = people_boxes[idx]['box']
#                     rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
#                                            linewidth=1.5, edgecolor='gray', facecolor='none')
#                     ax.add_patch(rect)
#                     plt.text(x1, y1-5, "Single", color='black', fontsize=8,
#                             bbox=dict(facecolor='white', alpha=0.7))

#                 # Draw each cluster with distinct colors
#                 for i, cluster in enumerate(clusters):
#                     color = colors[i]

#                     # Draw bounding boxes for members of this cluster
#                     for idx in cluster['members']:
#                         x1, y1, x2, y2 = people_boxes[idx]['box']
#                         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
#                                                linewidth=1.5, edgecolor=color, facecolor='none')
#                         ax.add_patch(rect)

#                     # Draw cluster boundary as a circle
#                     centroid = cluster['centroid']
#                     radius = cluster['radius']
#                     circle = patches.Circle(centroid, radius, color=color,
#                                          fill=False, linestyle='--', linewidth=2)
#                     ax.add_patch(circle)

#                     # Add label with cluster info
#                     crowd_text = " (CROWD)" if cluster['is_crowd'] else ""
#                     plt.text(centroid[0], centroid[1]-10,
#                             f"Cluster {cluster['id']+1}: {cluster['size']} people{crowd_text}",
#                             color='white', fontsize=10, weight='bold',
#                             bbox=dict(facecolor=color, alpha=0.7))

#             plt.text(50, 50, f"DBSCAN Parameters: eps={eps}, min_points_in_cluster={min_crowd_size}",
#                     fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

#             plt.title(f"Crowd Detection - {len(people_boxes)} people detected, {len(clusters)} clusters found")
#             plt.axis('off')

#             # Save and display the visualization
#             output_path = os.path.join("debug_output", f"live_result_{int(time.time())}.jpg")
#             plt.savefig(output_path, bbox_inches='tight', dpi=150)
#             plt.close()

#             # Display results
#             display(Image(output_path))
#             print(f"\nDetection results:")
#             print(f"- People detected: {len(people_boxes)}")
#             print(f"- Clusters found: {len(clusters)}")
#             if clusters:
#                 print(f"- Largest cluster: {max(c['size'] for c in clusters)} people")

#     except KeyboardInterrupt:
#         print("\nLive demo ended by user")
#     except Exception as e:
#         print(f"Error during live demo: {str(e)}")

# Example usage
image_folder = "/content/drive/MyDrive/groupB/tp/images1"
gt_folder = "/content/drive/MyDrive/groupB/tp/ground"

# Set these values as needed for your application
# EPS = 700  # Distance threshold for DBSCAN clustering
# MIN_CROWD_SIZE = 2  # Minimum number of people to define a crowd

# Use for testing a single image
test_image = './testpic.png'
# test_single_image(test_image)
# test_single_image_live1(eps=200, min_crowd_size=1)
# Uncomment to run the full dataset processing
# results_df = process_shanghaitech_dataset(
#     image_dir=image_folder,
#     gt_dir=gt_folder,
#     output_dir="shanghai_results",
#     max_people=50  # Maximum number of people to process in an image
# )


# Display some of the processed images in Colab
# import glob
# from IPython.display import Image, display

# # Get all result images
# result_images = glob.glob("shanghai_results/result_*.jpg")
# # Display first 5 (or however many you want)
# for img_path in result_images[:5]:
#     print(f"Displaying: {img_path}")
#     display(Image(img_path))


def process_single_image(image):
    """Process a single image (numpy array) and return results"""
    # Initialize YOLO model
    model = YOLO('yolov8l.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Detect people and clusters
    people_boxes, clusters = detect_people_and_clusters(model, image_rgb)

    # Create visualization
    plt.figure(figsize=(16, 12))
    plt.imshow(image_rgb)
    visualize_results(image_rgb, people_boxes, clusters, "temp_result.jpg")
    
    # Read the saved image
    with open("temp_result.jpg", "rb") as img_file:
        processed_image = img_file.read()
    
    return {
        'people_count': len(people_boxes),
        'cluster_count': len(clusters),
        'largest_cluster': max([c['size'] for c in clusters], default=0),
        'processed_image': processed_image
    }

def process_live_capture():
    """Capture and process a live image"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Failed to open camera"
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, "Failed to capture image"
    
    return process_single_image(frame), None