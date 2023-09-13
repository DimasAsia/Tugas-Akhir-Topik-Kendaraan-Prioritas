import cv2
import numpy as np
import threading
import time
import pygame
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

index_path = 0

def detect_objects(durasi, jalan_mana, ):
    # Variable global
    global line1, line2, line3, index_path
    
    
    cap1 = cv2.VideoCapture(video_path[index_path])
    cap2 = cv2.VideoCapture(video_path[index_path + 1])
    cap3 = cv2.VideoCapture(video_path[index_path + 2])
    cap4 = cv2.VideoCapture(video_path[index_path + 3])
    
    
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
    

    
    vehicle_counter_1 = 0
    vehicle_counter_2 = 0
    vehicle_counter_3 = 0
    vehicle_counter_4 = 0
    
    density_1 = 0
    density_2 = 0
    density_3 = 0
    density_4 = 0
    
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
                else:
                    green_light_duration_1 = fuzzy_traffic_control(density_1, vehicle_counter_1)
                     
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
                else:
                    green_light_duration_2 = fuzzy_traffic_control(density_2, vehicle_counter_2)
                
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
                                    emergency_detected_3 = True
                                
        
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
                    else:
                        
                        green_light_duration_3 = fuzzy_traffic_control(density_3, vehicle_counter_3)
                    
                    # Check if it's in emergency condition
                    if condition_3 == "Emergency":
                        # Set all traffic lights to green
                        green_light_darurat_3 += 1
                        green_light_duration_3 = green_light_darurat_3
                        status_jalan = 3
                        running_3 = False
                    else:
                        if not running_3:
                            condition = "Emergency"
                            vehicle_counter = vehicle_counter_3
                            density = density_3
                            green_light_duration = green_light_darurat_3
                            break
                    
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
                    else:
                        green_light_duration_4 = fuzzy_traffic_control(density_4, vehicle_counter_4)
                    
                    # Check if it's in emergency condition
                    if condition_4 == "Emergency":
                        # Set all traffic lights to green
                        green_light_darurat_4 += 1
                        
                        status_jalan = 4
                        running_4 = False
                    else:
                        if not running_4:
                            condition = "Emergency"
                            vehicle_counter = vehicle_counter_4
                            density = density_4
                            green_light_duration = green_light_darurat_4
                            break
                    
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
            # Check if the red light duration has 
            elapsed_time += 1
            if elapsed_time >= durasi:
                if jalan_mana == 1:
                    condition = "Normal"
                    vehicle_counter = vehicle_counter_2
                    density = density_2
                    green_light_duration = green_light_duration_2
                    status_jalan = 2
                    # Reset start time for the next video
                    elapsed_time = 0
                    
                    # Geser ke video berikutnya
                    break
                elif jalan_mana == 2:
                    condition = "Normal"
                    vehicle_counter = vehicle_counter_3
                    density = density_3
                    green_light_duration = green_light_duration_3
                    status_jalan = 3
                    # Reset start time for the next video
                    elapsed_time = 0
                    
                    # Geser ke video berikutnya
                    break
                if jalan_mana == 3:
                    condition = "Normal"
                    vehicle_counter = vehicle_counter_4
                    density = density_4
                    green_light_duration = green_light_duration_4
                    status_jalan = 4
                    # Reset start time for the next video
                    elapsed_time = 0
                    
                    # Geser ke video berikutnya
                    break
                if jalan_mana == 4:
                    condition = "Normal"
                    vehicle_counter = vehicle_counter_1
                    density = density_1
                    green_light_duration = green_light_duration_1
                    status_jalan = 1
                    # Reset start time for the next video
                    elapsed_time = 0
                    
                    # Geser ke video berikutnya
                    break
                
        
        index_path += 4
        if index_path == len(video_path):
            index_path = 0
        
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

def run_detection(durasi, jalan_mana):
    global density, condition, vehicle_counter, green_light_duration, status_jalan
    condition, density, vehicle_counter, green_light_duration, status_jalan = detect_objects(durasi, jalan_mana,)
    

def pygame_simulasi():
    pygame.init()
    width, height = 995, 650
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    background = pygame.image.load('images/intersection.png')
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    

    #inisialisasi awal
    jalan1 = 13
    jalan2 = '---'
    jalan3 = '---'
    jalan4 = '---'
    lampu_kuning = 3
    signal1 = redSignal
    signal2 = redSignal
    signal3 = redSignal
    signal4 = redSignal
    kepadatan = 'sepi'
    kondisi = 'Normal'
    status = 0
    jumlah = 5
    lampu_hijau = jalan1
    lampu_darurat = 0
    
    sudah_jalan1 = False
    sudah_jalan2 = False
    sudah_jalan3 = False
    sudah_jalan4 = False
    
    
    running = True
    
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        font = pygame.font.Font(None, 30)
        # Gambar garis-garis dan elemen lainnya di sini
        screen.fill((255, 255, 255))
        screen.blit(background,(0,0))
        
        
        pygame.draw.rect(screen, (0, 0, 0), (320, 187, 40, 40))
        text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
        screen.blit(text_jalan1, (330, 197))
        
        pygame.draw.rect(screen, (0, 0, 0), (315, 419, 40, 40))
        text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
        screen.blit(text_jalan2, (325, 430))
        
        pygame.draw.rect(screen, (0, 0, 0), (621, 419, 40, 40))
        text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
        screen.blit(text_jalan3, (630, 430))
        
        pygame.draw.rect(screen, (0, 0, 0), (611, 186, 40, 40))
        text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
        screen.blit(text_jalan4, (620, 197))
        
          
        screen.blit(signal1, (360, 138))
        screen.blit(signal2, (355, 418))
        screen.blit(signal3, (590, 418))
        screen.blit(signal4, (580, 138))
        
        text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
        screen.blit(text_density, (10, 20))
        text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
        screen.blit(text_condition, (10, 60))
        text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
        screen.blit(text_vehicle_counter, (10, 100))
        text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
        screen.blit(text_green_light_duration, (10, 140))
        

        pygame.display.flip()
        clock.tick(1)

        cek = False
        cek_alive = True
        perform_detection = True
        emergency_cek = True
        cek_lampu = True
        
        if type(jalan1) == int and type(jalan4) == str:
    
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            if perform_detection and type(jalan2) == str:
                sudah_jalan1 = True
                if jalan1 == 0 and status == 1:
                    jalan1 = '---'
                    jalan2 = 7
                    signal1 = redSignal
                    signal2 = greenSignal
                    screen.blit(signal1, (360, 138))
                    screen.blit(signal2, (355, 418))
                    text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                    screen.blit(text_jalan1, (330, 197))
                    
                else:
                    durasi = jalan1
                    jalan_mana = 1
                    # Start the detection thread
                    detection_thread = threading.Thread(target=run_detection, args=(durasi, jalan_mana,))
                    detection_thread.start()
                
                perform_detection = False
        
            
            if not cek and type(jalan2) == str:
                jalan2 = jalan1 + lampu_kuning
                signal1 = greenSignal
    
                cek = True    
            
            screen.blit(signal1, (360, 138))
            pygame.display.update()
            
            
            jalan1 -= 1
            text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
            screen.blit(text_jalan1, (330, 197))
                        
            jalan2 -= 1
            text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
            screen.blit(text_jalan2, (325, 430)) 
            
            if detection_thread.is_alive():
                test = 0
            else:
                cek_alive = False
                kondisi = condition
                if density == 0:
                    kepadatan = "kosong"
                elif density == 1:
                    kepadatan = "sepi"
                elif density == 2:
                    kepadatan = "sedang"
                elif density == 3:
                    kepadatan = "ramai"
                jumlah = vehicle_counter
                status = status_jalan
                    
                lampu_hijau = green_light_duration
            
            if kondisi == "Emergency" and not cek_alive:
                if cek_lampu:
                    jalan1 = 0
                    jalan2 = 0
                    cek_lampu = False
                    emergency_cek = False
                
                
            else:
                time.sleep(1)
                if jalan1 == 0 and jalan2 == 3:
                    jalan1 = lampu_kuning +1
                    jalan1 -= 1
                    signal1 = yellowSignal
                    screen.blit(signal1, (360, 138))
                
            
            
            if jalan1 == 0 and jalan2 == 0 and status == 2:
                jalan1 = '---'
                jalan2 = lampu_hijau
                if jalan2 == 0 and lampu_hijau == 0:
                    signal2 = redSignal
                else:
                    signal2 = greenSignal
                    
                signal1 = redSignal
                screen.blit(signal1, (360, 138))
                screen.blit(signal2, (355, 418))
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                
                
            elif jalan1 == 0 and jalan2 == 0 and status == 3:

                jalan1 = '---'
                jalan2 = '---'
                jalan3 = lampu_hijau
                signal1 = redSignal
                signal2 = redSignal
                signal3 = greenSignal
                screen.blit(signal1, (360, 138))
                screen.blit(signal2, (355, 418))
                screen.blit(signal3, (590, 418))
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
                
            elif jalan1 == 0 and jalan2 == 0 and status == 4:

                jalan1 = '---'
                jalan2 = '---'
                jalan4 = lampu_hijau
                signal1 = redSignal
                signal2 = redSignal
                signal4 = greenSignal
                screen.blit(signal1, (360, 138))
                screen.blit(signal2, (355, 418))
                screen.blit(signal4, (580, 138))
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
        
        elif type(jalan2) == int and type(jalan1) == str:
            
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            
            if perform_detection and type(jalan3) == str:
                if jalan2 == 0 and status == 2:
                    jalan2 = '---'
                    jalan3 = 7
                    signal2 = redSignal
                    signal3 = greenSignal
                    screen.blit(signal2, (355, 418))
                    screen.blit(signal3, (590, 418))
                    text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                    screen.blit(text_jalan2, (325, 430))
                    
                else:
                    durasi = jalan2
                    jalan_mana = 2
                    # Start the detection thread
                    detection_thread = threading.Thread(target=run_detection, args=(durasi, jalan_mana,))
                    detection_thread.start()
                
                sudah_jalan2 = True
                perform_detection = False

                
            if not cek and type(jalan3) == str:
                jalan3 = jalan2 + lampu_kuning
                cek = True
                
                
            jalan2 -= 1
            
            text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
            screen.blit(text_jalan2, (325, 430))
            
            jalan3 -= 1
            text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
            screen.blit(text_jalan3, (630, 430))
            
            if detection_thread.is_alive():
                test = 0
            else:
                kondisi = condition
                if density == 0:
                    kepadatan = "kosong"
                elif density == 1:
                    kepadatan = "sepi"
                elif density == 2:
                    kepadatan = "sedang"
                elif density == 3:
                    kepadatan = "ramai"
                jumlah = vehicle_counter
                status = status_jalan
                cek_alive = False
                    
                lampu_hijau = green_light_duration
            
            if kondisi == "Emergency" and not cek_alive:
                if cek_lampu:
                    jalan2 = 0
                    jalan3 = 0
                    cek_lampu = False
                    emergency_cek = False
                     
            else:
                time.sleep(1)
                if jalan2 == 0 and jalan3 == 3:
                    jalan2 = lampu_kuning +1
                    jalan2 -= 1
                    signal2 = yellowSignal
                    screen.blit(signal2, (355, 418))
            
                
            if jalan2 == 0 and jalan3 == 0 and status == 3:
                jalan2 = '---'
                jalan3 = lampu_hijau
                if jalan3 == 0 and lampu_hijau == 0:
                    signal3 = redSignal
                else:
                    signal3 = greenSignal
                    
                signal2 = redSignal
                screen.blit(signal2, (355, 418))
                screen.blit(signal3, (590, 418))
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
                
            elif jalan2 == 0 and jalan3 == 0 and status == 4:
                jalan3 = '---'
                jalan2 = '---'
                jalan4 = lampu_hijau
                signal2 = redSignal
                signal4 = greenSignal
                signal3 = redSignal
                screen.blit(signal2, (355, 418))
                screen.blit(signal3, (590, 418))
                screen.blit(signal4, (580, 138))
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (630, 430))
            
            elif jalan2 == 0 and jalan3 == 0 and status == 1:
                jalan3 = '---'
                jalan2 = '---'
                jalan1 = lampu_hijau
                signal2 = redSignal
                signal1 = greenSignal
                signal3 = redSignal
                screen.blit(signal2, (355, 418))
                screen.blit(signal3, (590, 418))
                screen.blit(signal1, (360, 138))
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (630, 430))
                
        elif type(jalan3) == int and type(jalan2) == str:
            
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            if perform_detection and type(jalan4) == str:
                sudah_jalan3 = True
                if jalan3 == 0 and status == 3:

                    jalan4 = 7
                    jalan3 = '---'
                    screen.blit(redSignal, (590, 418))
                    signal3 = redSignal
                    signal4 = greenSignal
                    text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                    screen.blit(signal4, (580, 138))
                    screen.blit(text_jalan3, (630, 430))
                
                else:
                    durasi = jalan3
                    jalan_mana = 3
                    # Start the detection thread
                    detection_thread = threading.Thread(target=run_detection, args=(durasi, jalan_mana))
                    detection_thread.start()
                
                perform_detection = False
            
            
            if not cek and type(jalan4) == str:
                jalan4 = jalan3 + lampu_kuning
                cek = True
            
            
            jalan3 -= 1
            
            text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
            screen.blit(text_jalan3, (630, 430))
            
            jalan4 -= 1
            text_merah = font.render(str(jalan4), True, (255, 255, 255))
            screen.blit(text_merah, (365, 130))
            
            if detection_thread.is_alive():
                test = 0
            else:
                
                kondisi = condition
                if density == 0:
                    kepadatan = "kosong"
                elif density == 1:
                    kepadatan = "sepi"
                elif density == 2:
                    kepadatan = "sedang"
                elif density == 3:
                    kepadatan = "ramai"
                jumlah = vehicle_counter
                status = status_jalan
                cek_alive = False
                    
                lampu_hijau = green_light_duration
            
            if kondisi == "Emergency" and not cek_alive:
                if cek_lampu:
                    jalan3 = 0
                    jalan4 = 0
                    cek_lampu = False
                    emergency_cek = False
                     
            else:
                time.sleep(1)
                if jalan3 == 0 and jalan4 == 3:
                    jalan3 = lampu_kuning +1
                    jalan3 -= 1
                    signal3 = yellowSignal
                    screen.blit(signal3, (590, 418))
                
             
            
            if jalan3 == 0 and jalan4 == 0 and status == 4:

                jalan4 = lampu_hijau
                if jalan4 == 0 and lampu_hijau == 0:
                    signal4 = redSignal
                else:
                    signal4 = greenSignal
                    
                jalan3 = '---'
                screen.blit(redSignal, (590, 418))
                signal3 = redSignal
                
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(signal4, (580, 138))
                screen.blit(text_jalan3, (630, 430))
            
            elif jalan3 == 0 and jalan4 == 0 and status == 1:
                jalan3 = '---'
                jalan4 = '---'
                jalan1 = lampu_hijau
                signal3 = redSignal
                signal1 = greenSignal
                signal4 = redSignal
                screen.blit(signal4, (580, 138))
                screen.blit(signal3, (590, 418))
                screen.blit(signal1, (360, 138))
                text_merah = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_merah, (365, 130))
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (630, 430))
            
            elif jalan3 == 0 and jalan4 == 0 and status == 2:
                jalan3 = '---'
                jalan4 = '---'
                jalan2 = lampu_hijau
                signal3 = redSignal
                signal2 = greenSignal
                signal4 = redSignal
                screen.blit(signal4, (580, 138))
                screen.blit(signal3, (590, 418))
                screen.blit(signal2, (355, 418))
                text_merah = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_merah, (365, 130))
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (630, 430))
        
        elif type(jalan4) == int and type(jalan3) == str:
            
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            if perform_detection and type(jalan1) == str:
                if jalan4 == 0 and status == 4:
                    print("oke")
                    jalan1 = 7
                else:
                    durasi = jalan4
                    jalan_mana = 4
                    # Start the detection thread
                    detection_thread = threading.Thread(target=run_detection, args=(durasi, jalan_mana))
                    detection_thread.start()
                    
                perform_detection = False
                
            if not cek and type(jalan1) == str:
                jalan1 = jalan4 + lampu_kuning
                cek = True
                  
            if jalan4 == 0 and status == 4:
                jalan4 = '---'
                signal4 = redSignal
                signal1 = greenSignal
                screen.blit(signal4, (580, 138))
                screen.blit(signal1, (360, 138))
                text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_jalan4, (620, 197))
            else:
                jalan4 -= 1
                               
                text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_jalan4, (620, 197))
                
                jalan1 -= 1
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                
                if detection_thread.is_alive():
                    test = 0
                else:
                    
                    kondisi = condition
                    if density == 0:
                        kepadatan = "kosong"
                    elif density == 1:
                        kepadatan = "sepi"
                    elif density == 2:
                        kepadatan = "sedang"
                    elif density == 3:
                        kepadatan = "ramai"
                    jumlah = vehicle_counter
                    status = status_jalan
                    cek_alive = False
                        
                    lampu_hijau = green_light_duration
                
                if kondisi == "Emergency" and not cek_alive:
                    if cek_lampu:
                        jalan4 = 0
                        jalan1 = 0
                        cek_lampu = False
                        emergency_cek = False
                         
                else:
                    time.sleep(1)
                    if jalan4 == 0 and jalan1 == 3:
                        jalan4 = lampu_kuning +1
                        jalan4 -= 1
                        signal4 = yellowSignal
                        screen.blit(signal4, (580, 138))

                    
                if jalan4 == 0 and jalan1 == 0 and status == 1:

                    jalan1 = lampu_hijau
                    if jalan1 == 0 and lampu_hijau == 0:
                        signal1 = redSignal
                    else:
                        signal1 = greenSignal
                    jalan4 = '---'
                    signal4 = redSignal
                    
                    screen.blit(signal4, (580, 138))
                    screen.blit(signal1, (360, 138))
                    text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                    screen.blit(text_jalan4, (620, 197))
                    
                elif jalan4 == 0 and jalan1 == 0 and status == 2:

                    jalan1 = '---'
                    jalan4 = '---'
                    jalan2 = lampu_hijau
                    signal2 = greenSignal
                    signal4 = redSignal
                    signal1 = redSignal
                    screen.blit(signal4, (580, 138))
                    screen.blit(signal1, (360, 138))
                    screen.blit(signal2, (355, 418))
                    text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                    screen.blit(text_jalan4, (620, 197))
                    text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                    screen.blit(text_jalan1, (330, 197))
                
                if jalan4 == 0 and jalan1 == 0 and status == 3:

                    jalan1 = '---'
                    jalan4 = '---'
                    jalan3 = lampu_hijau
                    signal3 = greenSignal
                    signal4 = redSignal
                    signal1 = redSignal
                    screen.blit(signal4, (580, 138))
                    screen.blit(signal1, (360, 138))
                    screen.blit(signal3, (590, 418))
                    text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                    screen.blit(text_jalan4, (620, 197))
                    text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                    screen.blit(text_jalan1, (330, 197))
                
    pygame.quit()

if __name__ == '__main__':  
    # Path to the input videos
    video_path = ['C:/Users/Acer/yolov4/video/video2.mp4',
                  'C:/Users/Acer/yolov4/video/video3.mp4', 
                  'C:/Users/Acer/yolov4/video/video1.mp4',
                  'C:/Users/Acer/yolov4/video/video4.mp4',
                  'C:/Users/Acer/yolov4/video/video2.mp4',
                  'C:/Users/Acer/yolov4/video/video3.mp4',
                  'C:/Users/Acer/yolov4/video/video2.mp4',
                  'C:/Users/Acer/yolov4/video/video6.mp4'
                  ]
    
    pygame_simulasi()
    
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
    
    