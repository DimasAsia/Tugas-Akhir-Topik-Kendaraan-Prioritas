# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:55:46 2023

@author: Acer
"""

import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load YOLOv4 configuration and weight files
net = cv2.dnn.readNet('/Users/Acer/yolov4/yolov4-tiny/yolov4-tiny-custom-kendaraan_final.weights', '/Users/Acer/yolov4/darknet/cfg/yolov4-tiny-custom-kendaraan.cfg')

# Load class names
with open('/Users/Acer/yolov4/classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set threshold values for object detection
conf_threshold = 0.5
nms_threshold = 0.4

# Open video file
video_path = 'C:/Users/Acer/yolov4/video/video2.mp4'  # Path to your input video
cap = cv2.VideoCapture(video_path)

# Define the three lines for density estimation
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line1 = [(0, int(frame_height * 0.6)), (frame_width, int(frame_height * 0.6))]  # Line 1 at 50% of the frame height
line2 = [(0, int(frame_height * 0.3)), (frame_width, int(frame_height * 0.3))]  # Line 2 at 30% of the frame height
line3 = [(0, int(frame_height * 0.1)), (frame_width, int(frame_height * 0.1))]  # Line 3 at 10% of the frame height

def point_below_line(point, line):
    x, y = point
    (x1, y1), (x2, y2) = line

    # Menghitung persamaan garis dengan rumus (y - y1) = m(x - x1)
    m = (y2 - y1) / (x2 - x1)

    # Menghitung nilai y yang seharusnya berada pada garis pada titik x yang diberikan
    y_expected = m * (x - x1) + y1

    # Jika nilai y aktual lebih besar dari y yang diharapkan, berarti titik berada di bawah garis
    return y > y_expected

def calculate_density(detected_objects):
    density = 0  # Menginisialisasi density sebagai angka 0 (Empty)
    num_detected_objects = len(detected_objects)
    
    if num_detected_objects == 0:
        # Jika tidak ada objek yang terdeteksi, density dianggap sebagai Empty (0)
        density = 0
    elif num_detected_objects > 0 and point_below_line(detected_objects[0], line1):
        # Jika minimal satu objek terdeteksi dan berada di bawah line1, density dianggap sebagai Low (1)
        density = 1
    elif num_detected_objects > 0 and point_below_line(detected_objects[0], line2):
        # Jika minimal satu objek terdeteksi dan berada di bawah line2, kita perlu memeriksa kondisi lebih lanjut untuk menentukan density
        if num_detected_objects > 1 and not point_below_line(detected_objects[1], line1):
            # Jika terdapat lebih dari satu objek dan objek kedua tidak berada di bawah line1, density dianggap sebagai Medium (2)
            density = 2
        else:
            # Jika kondisi di atas tidak terpenuhi, density dianggap sebagai Low (1)
            density = 1
    elif num_detected_objects > 0 and point_below_line(detected_objects[0], line3):
        # Jika minimal satu objek terdeteksi dan berada di bawah line3, kita perlu memeriksa kondisi lebih lanjut untuk menentukan density
        if num_detected_objects > 1 and not point_below_line(detected_objects[1], line2):
            # Jika terdapat lebih dari satu objek dan objek kedua tidak berada di bawah line2, density dianggap sebagai High (3)
            density = 3
        else:
            # Jika kondisi di atas tidak terpenuhi, density dianggap sebagai Medium (2)
            density = 2
    elif num_detected_objects > 0 and not point_below_line(detected_objects[0], line2):
        # Jika minimal satu objek terdeteksi dan tidak berada di bawah line2, kita perlu memeriksa kondisi lebih lanjut untuk menentukan density
        if num_detected_objects > 1 and not point_below_line(detected_objects[1], line1):
            # Jika terdapat lebih dari satu objek dan objek kedua tidak berada di bawah line1, density dianggap sebagai Low (1)
            density = 1
    
    return density



# Function to draw lines on the frame
def draw_lines(frame):
    cv2.line(frame, line1[0], line1[1], (0, 255, 0), 2)
    cv2.line(frame, line2[0], line2[1], (0, 255, 0), 2)
    cv2.line(frame, line3[0], line3[1], (0, 255, 0), 2)

    # Add text labels above the lines
    cv2.putText(frame, "Low", (line1[0][0], line1[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Medium", (line2[0][0], line2[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "High", (line3[0][0], line3[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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
    emergency_detected = False

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

    # Draw bounding boxes and labels on the frame
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

    # Draw lines on the frame
    draw_lines(frame)

    # Display density, vehicle count, and condition on the frame
    cv2.putText(frame, f"Density: {density}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicle Count: {vehicle_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Condition: {condition}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Check if video file is successfully opened
if not cap.isOpened():
   print("Error opening video file")
   exit()

# Read the first frame
ret, frame = cap.read()

# Create output video writer
#output_path = 'C:/Users/Acer/yolov4/output_video.mp4'  # Path to your output video
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fps = cap.get(cv2.CAP_PROP_FPS)
#frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frame by frame
while ret:
    # Detect objects and calculate density
    processed_frame = detect_objects(frame)

    # Write the processed frame to the output video file
    #out.write(processed_frame)

    # Display the processed frame
    cv2.imshow('Density Estimation', processed_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

def fuzzy_traffic_control(density, vehicle_counter, condition):
    # Fuzzy Membership Functions
    density = ctrl.Antecedent(np.arange(0, 101, 1), 'density')
    vehicle_counter = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_counter')
    lampu_hijau = ctrl.Consequent(np.arange(0, 16, 1), 'lampu_hijau')

    # Fuzzy Membership Functions for 'density'
    density['empty'] = fuzz.trimf(density.universe, [0, 0, 20])
    density['low'] = fuzz.trimf(density.universe, [0, 20, 40])
    density['medium'] = fuzz.trimf(density.universe, [20, 40, 60])
    density['high'] = fuzz.trimf(density.universe, [40, 60, 100])

    # Fuzzy Membership Functions for 'vehicle_counter'
    vehicle_counter['kosong'] = fuzz.trimf(vehicle_counter.universe, [0, 0, 10])
    vehicle_counter['sedikit'] = fuzz.trimf(vehicle_counter.universe, [0, 10, 20])
    vehicle_counter['banyak'] = fuzz.trimf(vehicle_counter.universe, [10, 20, 100])

    # Fuzzy Membership Functions for 'lampu_hijau'
    lampu_hijau['5_detik'] = fuzz.trimf(lampu_hijau.universe, [0, 0, 5])
    lampu_hijau['10_detik'] = fuzz.trimf(lampu_hijau.universe, [5, 10, 15])
    lampu_hijau['15_detik'] = fuzz.trimf(lampu_hijau.universe, [10, 15, 15])

    # Fuzzy Rules
    rule1 = ctrl.Rule(condition, lampu_hijau['15_detik'])
    rule2 = ctrl.Rule(density['empty'] & vehicle_counter['kosong'], lampu_hijau['0_detik'])
    rule3 = ctrl.Rule((density['low'] | density['medium']) & vehicle_counter['sedikit'], lampu_hijau['5_detik'])
    rule4 = ctrl.Rule(density['medium'] & vehicle_counter['sedang'], lampu_hijau['10_detik'])
    rule5 = ctrl.Rule((density['high'] | density['empty']) & vehicle_counter['banyak'], lampu_hijau['15_detik'])

    # Fuzzy Control System
    traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    traffic_sim = ctrl.ControlSystemSimulation(traffic_ctrl)

    # Set input values
    traffic_sim.input['density'] = density
    traffic_sim.input['vehicle_counter'] = vehicle_counter

    # Crunch the numbers
    traffic_sim.compute()

    # Get the output value
    lampu_hijau_value = traffic_sim.output['lampu_hijau']
    lampu_merah_value = 15 - lampu_hijau_value

    # Create a frame
    frame = "+" + "-" * 20 + "+"
    value_frame = "|" + " " * 20 + "|"

    # Print the frame
    print(frame)
    print(value_frame)

    # Print the values
    value_str = f"| Lampu Hijau: {lampu_hijau_value:.2f} detik |"
    print(value_str)
    value_str = f"| Lampu Merah: {lampu_merah_value:.2f} detik |"
    print(value_str)

    # Print the frame
    print(value_frame)
    print(frame)


# Release the video capture and writer objects
cap.release()
#out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

