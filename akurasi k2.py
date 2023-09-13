import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# Set the path to the folder containing the images
path_to_images = 'C:/Users/Acer/yolov4/test/image2'
ground_truth_folder = 'C:/Users/Acer/yolov4/test/truth_label2'
output_folder = 'C:/Users/Acer/yolov4/test/output_results'

# Load YOLOv4 configuration and weight files
net = cv2.dnn.readNet('/Users/Acer/yolov4/yolov4-tiny/yolov4-tiny-custom-kendaraan_final.weights', '/Users/Acer/yolov4/darknet/cfg/yolov4-tiny-custom-kendaraan.cfg')

# Define the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
classes = []
with open('/Users/Acer/yolov4/classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get a list of all image files in the folder
image_files = [file for file in os.listdir(path_to_images) if file.endswith('.jpg') or file.endswith('.png')]

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for image_file in image_files:
    image_path = os.path.join(path_to_images, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_file_path = os.path.join(ground_truth_folder, label_file)
    
    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape  # Get image dimensions
    
    # Detect objects using YOLOv4
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Read the label file
    ground_truth_data = []
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            class_id = int(line[0])
            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])
            
            class_name = classes[class_id]

            ground_truth_data.append((class_id, x, y, w, h))

    # Perform object detection on the image
    detected_objects = []
    class_ids = []
    confidences = []
    boxes = []
    test = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Minimum confidence threshold
                test += 1
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                bbox_width = int(detection[2] * width)
                bbox_height = int(detection[3] * height)
                x = int(center_x - bbox_width / 2)
                y = int(center_y - bbox_height / 2)
                class_name = classes[class_id]

                detected_objects.append({
                    'class_name': class_name,
                    'bbox': (x, y, bbox_width, bbox_height)
                })
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, bbox_width, bbox_height])
            

    # Perform non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    # Define color map for different classes
    color_map = {
        'ambulance': (0, 0, 255),    # Red
        'mobil': (0, 255, 0),    # Green
        'motor': (255, 0, 0),  # Blue
        'pemadam': (249, 180, 21),  # orange
        'truk': (0, 255, 255),  # cyan
    }

    # Draw bounding boxes and class labels on the image
    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        class_name = classes[class_id]
        confidence = confidences[i]
        label = f'{class_name}: {confidence:.2f}'
        
        # Get color for the bounding box based on class name
        color = color_map.get(class_name, (0, 0, 0))  # Default color is black
        
        # Draw bounding box and label using the retrieved color
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Ground truth classes for the image
    ground_truth_classes = [classes[class_id] for class_id, _, _, _, _ in ground_truth_data]

    # ...

    # Initialize dictionaries to store metrics for each class
    class_metrics = {class_name: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for class_name in classes}
    
    # Compare detected classes with ground truth classes to calculate TP, TN, FP, FN
    for class_id in range(len(classes)):
        class_name = classes[class_id]
    
        for i in range(len(boxes)):
            predicted_class_id = class_ids[i]
            predicted_class_name = classes[predicted_class_id]
            
            if predicted_class_name == class_name:
                if i in indices:  # Check if the bounding box is selected after NMS
                    if class_name in ground_truth_classes:
                        class_metrics[class_name]['tp'] += 1
                    else:
                        class_metrics[class_name]['fp'] += 1
                else:
                    if class_name in ground_truth_classes:
                        class_metrics[class_name]['fn'] += 1
                    else:
                        class_metrics[class_name]['tn'] += 1

    # Print or save the results to a single output file per image
    output_file_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_output.txt')
    with open(output_file_path, 'w') as f:
        f.write("Image: {}\n".format(image_file))
        for class_name, metrics in class_metrics.items():
            tp = metrics['tp']
            tn = metrics['tn']
            fp = metrics['fp']
            fn = metrics['fn']
            total_predictions = tp + tn + fp + fn
            accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0
            accuracy_percent = accuracy * 100


            f.write(f'Class: {class_name}\n')
            f.write(f'TP: {tp}\n')
            f.write(f'TN: {tn}\n')
            f.write(f'FP: {fp}\n')
            f.write(f'FN: {fn}\n')
            f.write(f'Accuracy: {accuracy_percent:.2f}%\n')
            f.write('-' * 20 + '\n')

print("Output files saved in the output folder.")