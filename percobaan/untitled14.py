import cv2
import numpy as np
import threading
import time
import pygame
import skfuzzy as fuzz
from skfuzzy import control as ctrl

video_path = ['C:/Users/Acer/yolov4/video/video1.mp4',
              'C:/Users/Acer/yolov4/video/video2.mp4',
              'C:/Users/Acer/yolov4/video/video3.mp4',
              'C:/Users/Acer/yolov4/video/video4.mp4',
              'C:/Users/Acer/yolov4/video/video5.mp4']

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
    cv2.putText(frame, "Low", (line1[0][0], line1[0][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Medium", (line2[0][0], line2[0][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "High", (line3[0][0], line3[0][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def fuzzy_traffic_control(density, vehicle_counter):
    # Fuzzy logic controller for determining the green light duration
    # based on the density and vehicle count.
    density_level = ctrl.Antecedent(np.arange(0, 3, 1), 'density')  # Mengubah range menjadi 0-3
    vehicle_count_level = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_count')
    green_light_duration = ctrl.Consequent(np.arange(0, 61, 1), 'green_light_duration')

    # Define membership functions for density
    density_level['Empty'] = fuzz.trimf(density_level.universe, [0, 0, 0.5])  # Empty (0)
    density_level['Low'] = fuzz.trimf(density_level.universe, [0, 0.5, 1.5])  # Low (1)
    density_level['Medium'] = fuzz.trimf(density_level.universe, [0.5, 1.5, 2.5])  # Medium (2)
    density_level['High'] = fuzz.trimf(density_level.universe, [1.5, 2.5, 3])  # High (3)


    # Define membership functions for vehicle count
    vehicle_count_level['few'] = fuzz.trimf(vehicle_count_level.universe, [0, 10, 11])
    vehicle_count_level['moderate'] = fuzz.trimf(vehicle_count_level.universe, [10, 11, 20])
    vehicle_count_level['many'] = fuzz.trimf(vehicle_count_level.universe, [20, 21, 100])

    # Define membership functions for green light duration
    green_light_duration['short'] = fuzz.trimf(green_light_duration.universe, [0, 0, 5])
    green_light_duration['medium'] = fuzz.trimf(green_light_duration.universe, [0, 5, 10])
    green_light_duration['long'] = fuzz.trimf(green_light_duration.universe, [5, 10, 15])

    # Define fuzzy rules
    
    rule1 = ctrl.Rule(density_level['Empty'] & vehicle_count_level['few'], green_light_duration['short'])
    rule2 = ctrl.Rule(density_level['Empty'] & vehicle_count_level['moderate'], green_light_duration['short'])
    rule3 = ctrl.Rule(density_level['Low'] & vehicle_count_level['few'], green_light_duration['short'])
    rule4 = ctrl.Rule(density_level['Low'] & vehicle_count_level['moderate'], green_light_duration['medium'])
    rule5 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['few'], green_light_duration['medium'])
    rule6 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['moderate'], green_light_duration['medium'])
    rule7 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['many'], green_light_duration['long'])
    rule8 = ctrl.Rule(density_level['High'] & vehicle_count_level['few'], green_light_duration['short'])
    rule9 = ctrl.Rule(density_level['High'] & vehicle_count_level['moderate'], green_light_duration['long'])
    rule10 = ctrl.Rule(density_level['High'] & vehicle_count_level['many'], green_light_duration['long'])

    # Create and simulate the fuzzy control system
    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
    traffic_controller = ctrl.ControlSystemSimulation(control_system)

    traffic_controller.input['density'] = density
    traffic_controller.input['vehicle_count'] = vehicle_counter

   # Crunch the numbers
    try:
        traffic_controller.compute()
        green_light_duration_output = round(traffic_controller.output['green_light_duration'])
    except Exception as e:
        green_light_duration_output = 0

    return green_light_duration_output

def detect_objects(durasi):
    # Variable global
    global line1, line2, line3
    
    cap1 = cv2.VideoCapture(video_path[0])
    cap2 = cv2.VideoCapture(video_path[1])
    cap3 = cv2.VideoCapture(video_path[2])
    cap4 = cv2.VideoCapture(video_path[3])
    
    
    # Menghitung lebar dan tinggi baru
    new_width = 360
    new_height = 360

    # Define the lines for density estimation
    line1 = [(0, int(new_height * 0.6)), (new_width, int(new_height * 0.6))]  # Line 1 at 60% of the frame height
    line2 = [(0, int(new_height * 0.3)), (new_width, int(new_height * 0.3))]  # Line 2 at 30% of the frame height
    line3 = [(0, int(new_height * 0.1)), (new_width, int(new_height * 0.1))]  # Line 3 at 10% of the frame height
    
    frame_count0 = 0
    frame_count1 = 0
    frame_count2 = 0
    frame_count3 = 0
    frame_count4 = 0
    frame_count5 = 0
    
    frame_interval = 30  # Deteksi dilakukan setiap 30 frame
    
    elapsed_time = 0
    
    green_light_duration_1 = 0
    green_light_duration_2 = 0
    green_light_duration_3 = 0
    green_light_duration_4 = 0
    
    green_light_darurat_1 = 0
    green_light_darurat_2 = 0
    green_light_darurat_3 = 0
    green_light_darurat_4 = 0
    
    running_1 = True
    running_2 = True
    running_3 = True
    running_4 = True
    
    condition = "Normal"
    vehicle_counter = 0
    density = 0
    green_light_duration = 0
    
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()

        if not (ret1 and ret2):
            break
        
        frame1 = cv2.resize(frame1, (new_width, new_height))
        frame2 = cv2.resize(frame2, (new_width, new_height))
        frame3 = cv2.resize(frame3, (new_width, new_height))
        frame4 = cv2.resize(frame4, (new_width, new_height))
        
        

        if (ret1):
            frame_count1 += 1

            # Deteksi dilakukan hanya pada frame yang memenuhi interval yang ditentukan
            if frame_count1 % frame_interval == 0:
                # Perform object detection
                blob = cv2.dnn.blobFromImage(frame1, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(output_layers)
                
                # Process the output layers
                class_ids = []
                confidences = []
                boxes = []
                detected_objects = []
                emergency_detected_1 = False
    
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
    
                        if confidence > conf_threshold:
                            center_x = int(detection[0] * frame1.shape[1])
                            center_y = int(detection[1] * frame1.shape[0])
                            width = int(detection[2] * frame1.shape[1])
                            height = int(detection[3] * frame1.shape[0])
    
                            # Calculate top-left corner coordinates of bounding box
                            top_left_x = int(center_x - (width / 2))
                            top_left_y = int(center_y - (height / 2))
    
    
                            # Add detected object to the list
                            detected_objects.append((center_x, center_y))
    
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([top_left_x, top_left_y, width, height])
                            
                            # Check if the detected object is an ambulance or fire truck
                            if classes[class_id] == 'ambulance' or classes[class_id] == 'pemadam':
                                emergency_detected_1 = True
                            
    
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
                vehicle_counter_1 = 0
                for i in indices:
                    x, y, w, h = boxes[i]
                    label = classes[class_ids[i]]
                    confidence = confidences[i]
                    color = color_map.get(label, (0, 0, 0))  # Get color from color_map, default is black
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame1, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Check if the detected object is a vehicle
                    if label == 'ambulance' or label == 'mobil' or label == 'motor' or label == 'pemadam' or label == 'truk':
                        vehicle_counter_1 += 1

                # Calculate density based on line intersections
                density_1 = calculate_density(detected_objects)
    
                # Determine the condition based on emergency detection
                condition_1 = "Normal"
                if emergency_detected_1:
                    condition_1 = "Emergency"
                    condition = "Emergency"
                else:
                    green_light_duration_1 = fuzzy_traffic_control(density_1, vehicle_counter_1)
                     
                
                # Add text to the frame indicating the green light duration
                cv2.putText(frame1, f"Density: {density_1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.putText(frame1, f"Vehicle Count: {vehicle_counter_1}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.putText(frame1, f"Condition: {condition_1}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.putText(frame1, f"Green Light: {green_light_duration_1} sec", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    
               
            
        if (ret2):
            frame_count2 += 1

            # Deteksi dilakukan hanya pada frame yang memenuhi interval yang ditentukan
            if frame_count2 % frame_interval == 0:
                # Perform object detection
                blob = cv2.dnn.blobFromImage(frame2, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(output_layers)
                
                # Process the output layers
                class_ids = []
                confidences = []
                boxes = []
                detected_objects = []
                emergency_detected_2 = False
    
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
    
                        if confidence > conf_threshold:
                            center_x = int(detection[0] * frame2.shape[1])
                            center_y = int(detection[1] * frame2.shape[0])
                            width = int(detection[2] * frame2.shape[1])
                            height = int(detection[3] * frame2.shape[0])
    
                            # Calculate top-left corner coordinates of bounding box
                            top_left_x = int(center_x - (width / 2))
                            top_left_y = int(center_y - (height / 2))
    
    
                            # Add detected object to the list
                            detected_objects.append((center_x, center_y))
    
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([top_left_x, top_left_y, width, height])
                            
                            # Check if the detected object is an ambulance or fire truck
                            if classes[class_id] == 'ambulance' or classes[class_id] == 'pemadam':
                                emergency_detected_2 = True
                            
    
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
                vehicle_counter_2 = 0
                for i in indices:
                    x, y, w, h = boxes[i]
                    label = classes[class_ids[i]]
                    confidence = confidences[i]
                    color = color_map.get(label, (0, 0, 0))  # Get color from color_map, default is black
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame2, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Check if the detected object is a vehicle
                    if label == 'ambulance' or label == 'mobil' or label == 'motor' or label == 'pemadam' or label == 'truk':
                        vehicle_counter_2 += 1

                # Calculate density based on line intersections
                density_2 = calculate_density(detected_objects)
    
                # Determine the condition based on emergency detection
                condition_2 = "Normal"
                if emergency_detected_2:
                    condition_2 = "Emergency"
                    condition = "Emergency"
                else:
                    green_light_duration_2 = fuzzy_traffic_control(density_2, vehicle_counter_2)
                
                # Add text to the frame indicating the green light duration
                cv2.putText(frame2, f"Density: {density_2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.putText(frame2, f"Vehicle Count: {vehicle_counter_2}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.putText(frame2, f"Condition: {condition_2}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.putText(frame2, f"Green Light: {green_light_duration_2} sec", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    
                
                
            if (ret3):
                frame_count3 += 1

                # Deteksi dilakukan hanya pada frame yang memenuhi interval yang ditentukan
                if frame_count3 % frame_interval == 0:
                    # Perform object detection
                    blob = cv2.dnn.blobFromImage(frame3, 1/255.0, (416, 416), swapRB=True, crop=False)
                    net.setInput(blob)
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                    outputs = net.forward(output_layers)
                    
                    # Process the output layers
                    class_ids = []
                    confidences = []
                    boxes = []
                    detected_objects = []
                    emergency_detected_3 = False
        
                    for output in outputs:
                        for detection in output:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
        
                            if confidence > conf_threshold:
                                center_x = int(detection[0] * frame3.shape[1])
                                center_y = int(detection[1] * frame3.shape[0])
                                width = int(detection[2] * frame3.shape[1])
                                height = int(detection[3] * frame3.shape[0])
        
                                # Calculate top-left corner coordinates of bounding box
                                top_left_x = int(center_x - (width / 2))
                                top_left_y = int(center_y - (height / 2))
        
        
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
                    vehicle_counter_3 = 0
                    for i in indices:
                        x, y, w, h = boxes[i]
                        label = classes[class_ids[i]]
                        confidence = confidences[i]
                        color = color_map.get(label, (0, 0, 0))  # Get color from color_map, default is black
                        cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame3, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Check if the detected object is a vehicle
                        if label == 'ambulance' or label == 'mobil' or label == 'motor' or label == 'pemadam' or label == 'truk':
                            vehicle_counter_3 += 1

                    # Calculate density based on line intersections
                    density_3 = calculate_density(detected_objects)
        
                    # Determine the condition based on emergency detection
                    condition_3 = "Normal"
                    if emergency_detected_3:
                        condition_3 = "Emergency"
                        condition = "Emergency"
                    else:
                        green_light_duration_3 = fuzzy_traffic_control(density_3, vehicle_counter_3)
                    
                    # Add text to the frame indicating the green light duration
                    cv2.putText(frame3, f"Density: {density_3}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    cv2.putText(frame3, f"Vehicle Count: {vehicle_counter_3}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    cv2.putText(frame3, f"Condition: {condition_3}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    cv2.putText(frame3, f"Green Light: {green_light_duration_3} sec", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                        
                    
                    
            if (ret4):
                frame_count4 += 1

                # Deteksi dilakukan hanya pada frame yang memenuhi interval yang ditentukan
                if frame_count4 % frame_interval == 0:
                    # Perform object detection
                    blob = cv2.dnn.blobFromImage(frame2, 1/255.0, (416, 416), swapRB=True, crop=False)
                    net.setInput(blob)
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                    outputs = net.forward(output_layers)
                    
                    # Process the output layers
                    class_ids = []
                    confidences = []
                    boxes = []
                    detected_objects = []
                    emergency_detected_4 = False
        
                    for output in outputs:
                        for detection in output:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
        
                            if confidence > conf_threshold:
                                center_x = int(detection[0] * frame4.shape[1])
                                center_y = int(detection[1] * frame4.shape[0])
                                width = int(detection[2] * frame4.shape[1])
                                height = int(detection[3] * frame4.shape[0])
        
                                # Calculate top-left corner coordinates of bounding box
                                top_left_x = int(center_x - (width / 2))
                                top_left_y = int(center_y - (height / 2))
        
        
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
                    vehicle_counter_4 = 0
                    for i in indices:
                        x, y, w, h = boxes[i]
                        label = classes[class_ids[i]]
                        confidence = confidences[i]
                        color = color_map.get(label, (0, 0, 0))  # Get color from color_map, default is black
                        cv2.rectangle(frame4, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame4, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Check if the detected object is a vehicle
                        if label == 'ambulance' or label == 'mobil' or label == 'motor' or label == 'pemadam' or label == 'truk':
                            vehicle_counter_4 += 1

                    # Calculate density based on line intersections
                    density_4 = calculate_density(detected_objects)
        
                    # Determine the condition based on emergency detection
                    condition_4 = "Normal"
                    if emergency_detected_4:
                        condition_4 = "Emergency"
                        condition = "Emergency"
                    else:
                        green_light_duration_4 = fuzzy_traffic_control(density_4, vehicle_counter_4)
                    
                    # Add text to the frame indicating the green light duration
                    cv2.putText(frame4, f"Density: {density_4}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    cv2.putText(frame4, f"Vehicle Count: {vehicle_counter_4}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    cv2.putText(frame4, f"Condition: {condition_4}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    cv2.putText(frame4, f"Green Light: {green_light_duration_4} sec", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                        
                    
            frame_count0 += 1
            
            # Deteksi dilakukan hanya pada frame yang memenuhi interval yang ditentukan
            if frame_count0 % frame_interval == 0:
                hori1 = np.concatenate((frame1, frame2), axis=1)
                hori2 = np.concatenate((frame3, frame4), axis=1)
                ver = np.concatenate((hori1, hori2), axis=0)
                combinate_frame = cv2.resize(ver, (630, 630))
                
                cv2.imshow('frame', combinate_frame)
                
        frame_count5 += 1
        
        # Deteksi dilakukan hanya pada frame yang memenuhi interval yang ditentukan
        if frame_count5 % frame_interval == 0:
            
            if condition == "Emergency":
                # Check if it's in emergency condition
                
                if condition_1 == "Emergency":
                    # Set all traffic lights to green
                    green_light_darurat_1 += 1
                    green_light_duration_1 = green_light_darurat_1
                    status_jalan = 1
                    running_1 = False
                else:
                    if not running_1 :
                        condition = "Emergency"
                        vehicle_counter = vehicle_counter_1
                        density = density_1
                        green_light_duration = green_light_darurat_1
                        break
                        
                            
                # Check if it's in emergency condition
                if condition_2 == "Emergency":
                    # Set all traffic lights to green
                    green_light_darurat_2 += 1
                    green_light_duration_2 = green_light_darurat_2
                    status_jalan = 2
                    running_2 = False
                else:
                    if not running_2:
                        condition = "Emergency"
                        vehicle_counter = vehicle_counter_2
                        density = density_2
                        green_light_duration = green_light_darurat_2
                        break
                            
                # Check if it's in emergency condition
                if condition_3 == "Emergency":
                    # Set all traffic lights to green
                    green_light_darurat_3 += 1
                    green_light_duration_ = green_light_darurat_3
                    status_jalan = 3
                    running_3 = False
                else:
                    if not running_3:
                        condition = "Emergency"
                        vehicle_counter = vehicle_counter_3
                        density = density_3
                        green_light_duration = green_light_darurat_3
                        break
                            
            
                # Check if it's in emergency condition
                if condition_4 == "Emergency":
                    # Set all traffic lights to green
                    green_light_darurat_4 += 1
                    green_light_duration_4 = green_light_darurat_4
                    status_jalan = 4
                    running_4 = False
                else:
                    if not running_4:
                        condition = "Emergency"
                        vehicle_counter = vehicle_counter_4
                        density = density_4
                        green_light_duration = green_light_darurat_4
                        break
                
            else:
                # Check if the red light duration has 
                elapsed_time += 1
                if elapsed_time >= durasi:
                    # Reset start time for the next video
                    
                    elapsed_time = 0
                    # Geser ke video berikutnya
                    break
         
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        
    # Release resources
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()
    
    return condition, density, vehicle_counter, green_light_duration, status_jalan

if __name__ == '__main__':  
    # Path to the input videos
    video_path = ['C:/Users/Acer/yolov4/video/video2.mp4',
                  'C:/Users/Acer/yolov4/video/video1.mp4', 
                  'C:/Users/Acer/yolov4/video/video3.mp4',
                  'C:/Users/Acer/yolov4/video/video4.mp4',
                  'C:/Users/Acer/yolov4/video/video5.mp4']
    
    condition, density, vehicle_counter, green_light_duration, status_jalan = detect_objects(13)
    print(condition)
    print(density)
    print(vehicle_counter)
    print(green_light_duration)
    print(status_jalan)
    
    '''frame1 = detect_objects(cap1)
    frame2 = detect_objects(cap2)
    
    
    
    # Menggabungkan frame
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))
    combined_frame = np.vstack((top_row, bottom_row))
    
    # Display the resulting frame
    cv2.imshow('Frame', combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break'''
    
    