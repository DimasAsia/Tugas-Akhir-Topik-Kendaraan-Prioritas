# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:49:50 2023

@author: Acer
"""

import cv2
import numpy as np

# Load YOLOv4 configuration and weight files
net = cv2.dnn.readNet('/Users/Acer/yolov4/yolov4-tiny/yolov4-tiny-custom-kendaraan_final.weights', '/Users/Acer/yolov4/darknet/cfg/yolov4-tiny-custom-kendaraan.cfg')

# Load class names
with open('/Users/Acer/yolov4/classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set threshold values for object detection
conf_threshold = 0.5
nms_threshold = 0.4

# Function to detect objects and calculate density
def detect_objects(frame):
    height, width, _ = frame.shape

    # Create a blob from the input frame and perform a forward pass of YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []

    # Process each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                # Get the coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Store the bounding box information
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # Store the center point of the bounding box
                detected_objects.append((center_x, center_y))

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Define color map for different classes
    color_map = {
        'ambulance': (0, 0, 255),    # Red
        'mobil': (0, 255, 0),    # Green
        'motor': (255, 0, 0),  # Blue
        'pemadam': (249, 180, 21),  # Orange
        'truk': (0, 255, 255),  # Cyan
    }

    # Draw bounding boxes and labels on the frame
    for i in indices:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = color_map.get(label, (0, 0, 0))  # Get color from color_map, default is black
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
