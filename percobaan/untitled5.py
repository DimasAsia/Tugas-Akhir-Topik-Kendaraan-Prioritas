import cv2
import numpy as np
import time
import pygame
import threading
import winsound
import pygame
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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

def fuzzy_traffic_control(density, vehicle_counter):
    # Fuzzy logic controller for determining the green light duration
    # based on the density and vehicle count.
    density_level = ctrl.Antecedent(np.arange(0, 4, 1), 'density')  # Mengubah range menjadi 0-3
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
    rule2 = ctrl.Rule(density_level['Low'] & vehicle_count_level['few'], green_light_duration['short'])
    rule3 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['few'], green_light_duration['medium'])
    rule4 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['moderate'], green_light_duration['medium'])
    rule5 = ctrl.Rule(density_level['High'] & vehicle_count_level['moderate'], green_light_duration['long'])
    rule6 = ctrl.Rule(density_level['High'] & vehicle_count_level['many'], green_light_duration['long'])

    # Create and simulate the fuzzy control system
    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    traffic_controller = ctrl.ControlSystemSimulation(control_system)

    traffic_controller.input['density'] = density
    traffic_controller.input['vehicle_count'] = vehicle_counter

    # Crunch the numbers
    try:
        traffic_controller.compute()
        green_light_duration_output = round(traffic_controller.output['green_light_duration'])
    except:
        green_light_duration_output = 0

    return green_light_duration_output

# Function to draw lines on the frame
def draw_lines(frame):
    cv2.line(frame, line1[0], line1[1], (0, 255, 0), 2)
    cv2.line(frame, line2[0], line2[1], (0, 255, 0), 2)
    cv2.line(frame, line3[0], line3[1], (0, 255, 0), 2)

    # Add text labels above the lines
    cv2.putText(frame, "Low", (line1[0][0], line1[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Medium", (line2[0][0], line2[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "High", (line3[0][0], line3[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


import cv2
import numpy as np
import time


def detect_objects(image, net, classes, conf_threshold, nms_threshold):
    # variable global
    global line1, line2, line3
    
    # Get video width and height
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    
    
    while True:
        
        # Perform object detection
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
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
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            
                
        
        # Display the resulting frame
        cv2.imshow('Frame', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
def main():
    # Load YOLOv4 configuration and weight files
    net = cv2.dnn.readNet('/Users/Acer/yolov4/yolov4-tiny/yolov4-tiny-custom-kendaraan_final.weights', '/Users/Acer/yolov4/darknet/cfg/yolov4-tiny-custom-kendaraan.cfg')

    # Load class names
    with open('/Users/Acer/yolov4/classes.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Set parameters
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Load videos
    cap1 = cv2.VideoCapture('C:/Users/Acer/yolov4/video/video1.mp4')
    cap2 = cv2.VideoCapture('C:/Users/Acer/yolov4/video/video2.mp4')
    cap3 = cv2.VideoCapture('C:/Users/Acer/yolov4/video/video3.mp4')
    cap4 = cv2.VideoCapture('C:/Users/Acer/yolov4/video/video4.mp4')

    while True:
        # Read frames from videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
    
        if not (ret1 and ret2 and ret3 and ret4):
            break
    
        # Resize frames (optional)
        frame1 = cv2.resize(frame1, (300, 300))
        frame2 = cv2.resize(frame2, (300, 300))
        frame3 = cv2.resize(frame3, (300, 300))
        frame4 = cv2.resize(frame4, (300, 300))
    
        # Combine frames
        top_row = np.hstack((frame1, frame2))
        bottom_row = np.hstack((frame3, frame4))
        combined_frame = np.vstack((top_row, bottom_row))
    
        # Perform object detection on the combined frame
        detect_objects(combined_frame, net, classes, conf_threshold, nms_threshold)
    
        # Display the frame
        cv2.imshow("Multi-Video Detection", combined_frame)
    
        # Check if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # Release resources
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
