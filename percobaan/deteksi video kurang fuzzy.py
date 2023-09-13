# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:55:46 2023
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

# Define the three lines for density estimation
frame_width = 0
frame_height = 0

line1 = []
line2 = []
line3 = []

def point_below_line(point, line):
    x, y = point
    (x1, y1), (x2, y2) = line

    # Calculate the equation of the line: (y - y1) = m(x - x1)
    m = (y2 - y1) / (x2 - x1)

    # Calculate the expected y-value on the line at the given x-coordinate
    y_expected = m * (x - x1) + y1

    # If the actual y-value is greater than the expected y-value, the point is below the line
    return y > y_expected

def calculate_density(detected_objects):
    density = "Empty"
    num_detected_objects = len(detected_objects)
    
    if num_detected_objects == 0:
        # If no objects are detected, the density is considered empty.
        density = "Empty"
    elif num_detected_objects > 0 and point_below_line(detected_objects[0], line1):
        # If at least one object is detected and it is below line1,
        # the density is considered low.
        density = "Low"
    elif num_detected_objects > 0 and point_below_line(detected_objects[0], line2):
        # If at least one object is detected and it is below line2,
        # we need to check further conditions to determine the density.
        if num_detected_objects > 1 and not point_below_line(detected_objects[1], line1):
            # If there are multiple objects and the second object is not below line1,
            # the density is considered medium.
            density = "Medium"
        else:
            # If the above condition is not met, the density is considered low.
            density = "Low"
    elif num_detected_objects > 0 and point_below_line(detected_objects[0], line3):
        # If at least one object is detected and it is below line3,
        # we need to check further conditions to determine the density.
        if num_detected_objects > 1 and not point_below_line(detected_objects[1], line2):
            # If there are multiple objects and the second object is not below line2,
            # the density is considered high.
            density = "High"
        else:
            # If the above condition is not met, the density is considered medium.
            density = "Medium"
    elif num_detected_objects > 0 and not point_below_line(detected_objects[0], line2):
        # If at least one object is detected and it is not below line2,
        # the density is considered high.
        density = "High"
    
    return density

# process video
def detect_objects(cap):
    # variable global
    global line1, line2, line3
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate the new width and height
    new_width = int(frame_width * 0.6)
    new_height = int(frame_height * 0.6)
   

    # Define the lines for density estimation
    line1 = [(0, int(new_height * 0.6)), (new_width, int(new_height * 0.6))]  # Line 1 at 50% of the frame height
    line2 = [(0, int(new_height * 0.3)), (new_width, int(new_height * 0.3))]  # Line 2 at 30% of the frame height
    line3 = [(0, int(new_height * 0.1)), (new_width, int(new_height * 0.1))]  # Line 3 at 10% of the frame height

    while True:
        # Read the current frame
        ret, frame = cap.read()

        if not ret:
            break
        
        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Perform object detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        # Process the output layers
        class_ids = []
        confidences = []
        boxes = []
        detected_objects = []
        emergency_detected = False

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)

                    # Calculate top-left corner coordinates of bounding box
                    top_left_x = int(center_x - width / 2)
                    top_left_y = int(center_y - height / 2)

                    # Add detected object to the list
                    detected_objects.append((center_x, center_y))

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([top_left_x, top_left_y, width, height])
                    
                    # Check if the detected object is an ambulance or fire truck
                    if classes[class_id] == 'ambulance' or classes[class_id] == 'pemadam':
                        emergency_detected = True

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
        
        # Draw bounding boxes and labels
        vehicle_counter = 0
        for i in indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = color_map.get(label, (0, 0, 0))  # Get color from color_map, default is black
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check if the detected object is a vehicle
            if label == 'ambulance' or label == 'mobil' or label == 'motor' or label == 'pemadam' or label == 'truk':
                vehicle_counter += 1
                
        # Calculate density based on line intersections
        density = calculate_density(detected_objects)

        # Determine the condition based on emergency detection
        condition = "Normal"
        if emergency_detected:
            condition = "Emergency"
        
        # Draw lines for density estimation
        cv2.line(frame, line1[0], line1[1], (0, 255, 0), 2)
        cv2.line(frame, line2[0], line2[1], (0, 255, 0), 2)
        cv2.line(frame, line3[0], line3[1], (0, 255, 0), 2)

        # Add text labels above the lines
        cv2.putText(frame, "Low", (line1[0][0], line1[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Medium", (line2[0][0], line2[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "High", (line3[0][0], line3[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display density, vehicle count, and condition on the frame
        cv2.putText(frame, f"Density: {density}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicle Count: {vehicle_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Condition: {condition}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    


video_path = ['C:/Users/Acer/yolov4/video/video1.mp4',
              'C:/Users/Acer/yolov4/video/video2.mp4',
              'C:/Users/Acer/yolov4/video/video3.mp4',
              'C:/Users/Acer/yolov4/video/video4.mp4',
              'C:/Users/Acer/yolov4/video/video5.mp4',
              'C:/Users/Acer/yolov4/video/video6.mp4',
              'C:/Users/Acer/yolov4/video/video7.mp4',
              'C:/Users/Acer/yolov4/video/video8.mp4']


