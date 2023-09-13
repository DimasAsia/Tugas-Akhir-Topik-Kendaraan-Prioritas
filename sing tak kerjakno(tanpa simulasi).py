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
    vehicle_count_level['few'] = fuzz.trimf(vehicle_count_level.universe, [0, 5, 5])
    vehicle_count_level['moderate'] = fuzz.trimf(vehicle_count_level.universe, [5, 10, 15])
    vehicle_count_level['many'] = fuzz.trimf(vehicle_count_level.universe, [15, 21, 100])

    # Define membership functions for green light duration
    green_light_duration['short'] = fuzz.trimf(green_light_duration.universe, [0, 0, 5])
    green_light_duration['medium'] = fuzz.trimf(green_light_duration.universe, [6, 8, 10])
    green_light_duration['long'] = fuzz.trimf(green_light_duration.universe, [11, 20, 20])

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


# Process video
def detect_objects(video_path):
    # variable global
    global line1, line2, line3
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    # Define the lines for density estimation
    line1 = [(0, int(frame_height * 0.6)), (frame_width, int(frame_height * 0.6))]  # Line 1 at 60% of the frame height
    line2 = [(0, int(frame_height * 0.3)), (frame_width, int(frame_height * 0.3))]  # Line 2 at 30% of the frame height
    line3 = [(0, int(frame_height * 0.1)), (frame_width, int(frame_height * 0.1))]  # Line 3 at 10% of the frame height

    while True:
        # Read the current frame
        ret, frame = cap.read()

        if not ret:
            break

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
       
        
        # Draw bounding boxes and labels
        vehicle_counter = 0
        for i in indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]


            # Check if the detected object is a vehicle
            if label == 'ambulance' or label == 'mobil' or label == 'motor' or label == 'pemadam' or label == 'truk':
                vehicle_counter += 1

        # Calculate density based on line intersections
        density = calculate_density(detected_objects)

        # Determine the condition based on emergency detection
        condition = "Normal"
        if emergency_detected:
            condition = "Emergency"

        # Check if it's in emergency condition
        if condition == "Emergency":
            # Set all traffic lights to green
            green_light_duration = 1000  # A large value to keep the green light on

        else:
            # Calculate the green light duration using fuzzy logic
            green_light_duration = fuzzy_traffic_control(density, vehicle_counter)

        draw_lines(frame)

        # Add text to the frame indicating the green light duration
        cv2.putText(frame, f"Density: {density}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicle Count: {vehicle_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Condition: {condition}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Green Light: {green_light_duration} sec", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame to the output video file
        #out.write(frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    #out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Path to the input videos
video_paths = ['C:/Users/Acer/yolov4/video/video4.mp4']

# Generate output video paths
#output_paths = [video_path.replace('.mp4', '_output.mp4') for video_path in video_paths]


detect_objects('C:/Users/Acer/yolov4/video/video2.mp4')

'''# Loop over each video
for video_path, output_path in zip(video_paths, output_paths):
    detect_objects(video_path, output_path)'''
