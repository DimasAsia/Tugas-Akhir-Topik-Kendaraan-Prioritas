import os
import cv2
import numpy as np
import random
import glob
from object_deteksi import detect_objects
from sklearn.metrics import confusion_matrix, accuracy_score

# Set the path to the folder containing the images
path_to_images = 'C:/Users/Acer/yolov4/test/image2'
ground_truth_folder = 'C:/Users/Acer/yolov4/test/truth_label2'

# Load YOLOv4 configuration and weight files
net = cv2.dnn.readNet('/Users/Acer/yolov4/yolov4-tiny/yolov4-tiny-custom-kendaraan_final.weights', '/Users/Acer/yolov4/darknet/cfg/yolov4-tiny-custom-kendaraan.cfg')

# Define the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
classes = []
with open('/Users/Acer/yolov4/classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Select a random image from the folder
image_files = os.listdir(path_to_images)
random_image_file = random.choice(image_files)
random_image_path = os.path.join(path_to_images, random_image_file)

# Load image
image = cv2.imread(random_image_path)
height, width, _ = image.shape  # Get image dimensions

# Get the corresponding label file
label_file = os.path.splitext(random_image_file)[0] + '.txt'
label_file_path = os.path.join(ground_truth_folder, label_file)


# Detect objects using YOLOv4
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Rest of the code...

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

        ground_truth_data.append((class_id, x, y, w, h))

# Perform object detection on the image
detected_objects = []
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:  # Minimum confidence threshold
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


# Create empty lists for y_true and y_pred
y_true = []
y_pred = []

# Iterate over the ground truth data
for obj in ground_truth_data:
    class_id = obj[0]
    class_name = classes[class_id]
    y_true.append(class_name)

    # Check if the class name exists in detected_objects
    if any(detection['class_name'] == class_name for detection in detected_objects):
        y_pred.append(class_name)
    else:
        y_pred.append('background')

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)

# Initialize a dictionary for per-class statistics
class_stats = {}

# Calculate TP, TN, FP, FN, and accuracy per class
for i, class_name in enumerate(classes):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    tn = np.sum(cm) - tp
    "accuracy = (tp + tn) / (tp + tn + fp + fn)"
    # Calculate Specificity
    "specificity = tn / (tn + fp)"

    class_stats[class_name] = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        #'specificity':specificity
        #'Accuracy': accuracy
        
    }



# Print per-class statistics
for class_name, stats in class_stats.items():
    print(f'{class_name}:')
    print(f'TP: {stats["TP"]}')
    print(f'TN: {stats["TN"]}')
    print(f'FP: {stats["FP"]}')
    print(f'FN: {stats["FN"]}')
    #print(f'FN: {stats["specificity"]}')
    #print(f'Accuracy: {stats["Accuracy"]}')
    print()

# Calculate overall accuracy
overall_accuracy = accuracy_score(y_true, y_pred)

print(f'Confusion Matriks:')
print(f'{cm}')

# Print overall accuracy
print(f'Accuracy: {overall_accuracy:.2f}')

# Display the resulting image
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
