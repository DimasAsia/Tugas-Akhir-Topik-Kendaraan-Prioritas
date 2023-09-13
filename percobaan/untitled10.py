import cv2
import numpy as np
import time
import pygame
import cv2
import threading
import winsound
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



# Process video
def detect_objects(video_path, durasi):
    # variable global
    global line1, line2, line3
    
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / frame_rate


    # Get video width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    '''# Define the output codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))'''

    # Define the lines for density estimation
    line1 = [(0, int(frame_height * 0.6)), (frame_width, int(frame_height * 0.6))]  # Line 1 at 60% of the frame height
    line2 = [(0, int(frame_height * 0.3)), (frame_width, int(frame_height * 0.3))]  # Line 2 at 30% of the frame height
    line3 = [(0, int(frame_height * 0.1)), (frame_width, int(frame_height * 0.1))]  # Line 3 at 10% of the frame height

    # Initialize red light duration and start time
    
    density = 0
    condition = 'normal'
    vehicle_counter = 0
    elapsed_time = 0
    green_light_duration = 0
    green_light_darurat = 0
    running = True
    

    while elapsed_time < duration:
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

        # Draw lines on the frame
        'draw_lines(frame)'
        
        # Check if it's in emergency condition
        if condition == "Emergency":
            # Set all traffic lights to green
            green_light_darurat += 1 /frame_rate
            green_light_duration = int(green_light_darurat)
            running = False
        else:
            if not running :
                if green_light_darurat > 5:
                    condition = "Emergency"
                    break
                    
                else:
                    green_light_duration += 3
                    condition = "Emergency"
                    break
                    
            else:
                green_light_duration = fuzzy_traffic_control(density, vehicle_counter)
                
            # Check if the red light duration has elapsed
            elapsed_time += 1 / frame_rate
            if elapsed_time >= durasi:
                # Reset start time for the next video
                elapsed_time = 0
                # Geser ke video berikutnya
                break
            
            time.sleep(1/frame_rate)

        # Add text to the frame indicating the green light duration
        cv2.putText(frame, f"Density: {density}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicle Count: {vehicle_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Condition: {condition}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Green Light: {green_light_duration} sec", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        '''# Write the frame to the output video file
        out.write(frame)'''

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release the video capture and writer objects
    cap.release()
    'out.release()'

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    return density, condition, vehicle_counter, green_light_duration

def run_detection(video_path, durasi):
    global density, condition, vehicle_counter, green_light_duration
    density, condition, vehicle_counter, green_light_duration = detect_objects(video_path, durasi)
    

if __name__ == '__main__':
    
    # Path to the input videos
    video_path = ['C:/Users/Acer/yolov4/video/video1.mp4',
                  'C:/Users/Acer/yolov4/video/video2.mp4', 
                  'C:/Users/Acer/yolov4/video/video3.mp4',
                  'C:/Users/Acer/yolov4/video/video4.mp4',
                  'C:/Users/Acer/yolov4/video/video5.mp4']
       
    
    pygame.init()
    width, height = 995, 650
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    background = pygame.image.load('images/intersection.png')
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    

    #inisialisasi awal
    jalan1 = 7
    jalan2 = '---'
    jalan3 = '---'
    jalan4 = '---'
    lampu_kuning = 5
    signal1 = redSignal
    signal2 = redSignal
    signal3 = redSignal
    signal4 = redSignal
    kepadatan = 'sepi'
    kondisi = 'Normal'
    jumlah = 5
    lampu_hijau = jalan1
    lampu_darurat = 0
    frequency = 1000


    running = True
    index_path = 0
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
        perform_detection = True
        
        
        if type(jalan1) == int and jalan1 > 0 and type(jalan4) == str:
            
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            if perform_detection and type(jalan2) == str:
                durasi = jalan1
                # Start the detection thread
                detection_thread = threading.Thread(target=run_detection, args=(video_path[index_path], durasi))
                detection_thread.start()
                perform_detection = False
                
            if kondisi == "Emergency":
                winsound.Beep(frequency, durasi)
            
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
            
            
            
            if jalan1 == 0 and jalan2 == 5:
                jalan1 = lampu_kuning +1
                jalan1 -= 1
                signal1 = yellowSignal
                screen.blit(signal1, (360, 138))
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                
                
            if jalan1 == 0 and jalan2 == 0:
                index_path += 1
                jalan2 = green_light_duration
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
                
                    
                lampu_hijau = green_light_duration
                lampu_darurat = green_light_duration
                jalan1 = '---'
                signal1 = redSignal
                signal2 = greenSignal
                screen.blit(redSignal, (360, 138))
                screen.blit(signal2, (355, 418))
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                    
            
            time.sleep(35/durasi)
                    
        elif type(jalan2) == int and jalan2 > 0 and type(jalan1) == str:
            
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            
            
            if perform_detection and type(jalan3) == str:
                durasi = jalan2
                # Start the detection thread
                detection_thread = threading.Thread(target=run_detection, args=(video_path[index_path], durasi))
                detection_thread.start()

                perform_detection = False
            
            if kondisi == "Emergency":
                winsound.Beep(frequency, durasi)
                
            if not cek and type(jalan3) == str:
                jalan3 = jalan2 + lampu_kuning
                cek = True
                  
            jalan2 -= 1
            
            text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
            screen.blit(text_jalan2, (325, 430))
            
            jalan3 -= 1
            text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
            screen.blit(text_jalan3, (630, 430))
            
            if jalan2 == 0 and jalan3 == 5:
                
                jalan2 = lampu_kuning +1
                jalan2 -= 1
                signal2 = yellowSignal
                screen.blit(signal2, (355, 418))
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
                
            if jalan2 == 0 and jalan3 == 0:
                index_path += 1
                jalan3 = green_light_duration
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
                
                    
                lampu_hijau = green_light_duration
                lampu_darurat = green_light_duration
                jalan2 = '---'
                signal2 = redSignal
                signal3 = greenSignal
                screen.blit(signal2, (355, 418))
                screen.blit(signal3, (590, 418))
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
            
            time.sleep(35/durasi)
            
        elif type(jalan3) == int and jalan3 > 0 and type(jalan2) == str:
            
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            if perform_detection and type(jalan4) == str:
                durasi = jalan3
                # Start the detection thread
                detection_thread = threading.Thread(target=run_detection, args=(video_path[index_path], durasi))
                detection_thread.start()
                perform_detection = False
            
            if kondisi == "Emergency":
                winsound.Beep(frequency, durasi)
            
            if not cek and type(jalan4) == str:
                jalan4 = jalan3 + lampu_kuning
                cek = True
                  
            jalan3 -= 1
            
            text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
            screen.blit(text_jalan3, (630, 430))
            
            jalan4 -= 1
            text_merah = font.render(str(jalan4), True, (255, 255, 255))
            screen.blit(text_merah, (365, 130))
            
            if jalan3 == 0 and jalan4 == 5:
                
                jalan3 = lampu_kuning +1
                jalan3 -= 1
                signal3 = yellowSignal
                screen.blit(signal3, (590, 418))
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (630, 430))
                
            if jalan3 == 0 and jalan4 == 0:
                index_path += 1
                jalan4 = green_light_duration
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
                lampu_hijau = green_light_duration
                jalan3 = '---'
                screen.blit(redSignal, (590, 418))
                signal3 = redSignal
                signal4 = greenSignal
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(signal4, (580, 138))
                screen.blit(text_jalan3, (630, 430))
            
            time.sleep(35/durasi)

        elif type(jalan4) == int and jalan4 > 0 and type(jalan3) == str:
            
            text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (10, 100))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (10, 140))
            
            if perform_detection and type(jalan1) == str:
                durasi = jalan4
                # Start the detection thread
                detection_thread = threading.Thread(target=run_detection, args=(video_path[index_path], durasi))
                detection_thread.start()
                perform_detection = False
                
            if not cek and type(jalan1) == str:
                jalan1 = jalan4 + lampu_kuning
                cek = True
                  
            jalan4 -= 1
                           
            text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
            screen.blit(text_jalan4, (620, 197))
            
            jalan1 -= 1
            text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
            screen.blit(text_jalan1, (330, 197))
            
            if jalan4 == 0 and jalan1 == 5:
                
                jalan4 = lampu_kuning +1
                jalan4 -= 1
                signal4 = yellowSignal
                screen.blit(signal4, (580, 138))
                text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_jalan4, (620, 197))
                
            if jalan4 == 0 and jalan1 == 0:
                index_path += 1
                jalan1 = green_light_duration
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
                lampu_hijau = green_light_duration
                jalan4 = '---'
                signal4 = redSignal
                signal1 = greenSignal
                screen.blit(signal4, (580, 138))
                screen.blit(signal1, (360, 138))
                text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_jalan4, (620, 197))
            
            time.sleep(35/durasi)
        
        if index_path == len(video_path):
            index_path = 0
                
    pygame.quit()





