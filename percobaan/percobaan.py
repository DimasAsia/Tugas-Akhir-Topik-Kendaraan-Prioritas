import cv2
import numpy as np
import threading
import time
import pygame

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




def detect_objects():
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
    
    pygame.init()
    width, height = 995, 650
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    background = pygame.image.load('images/intersection.png')
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    
    #inisialisasi awal
    jalan1 = 5
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
    jumlah = 5
    lampu_hijau = jalan1
    lampu_darurat = 0
    frame_counter = 0
    
    wadah_jumlah = 0
    
    frame_count1 = 0
    frame_count2 = 0
    
    frame_interval = 30  # Deteksi dilakukan setiap 30 frame
    
    
    perform_detection2 = True
    
    cek_alive = True
    cek_lampu = True
    running = True
    index_path = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            break
        
        frame1 = cv2.resize(frame1, (new_width, new_height))
        frame2 = cv2.resize(frame2, (new_width, new_height))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
                emergency_detected = False
    
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
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame1, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
                # Determine the condition based on emergency detection
                condition = "Normal"
                if emergency_detected:
                    condition = "Emergency"
                    
                # Display the resulting frame
                cv2.imshow('jalan1', frame1)
            
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
                emergency_detected = False
    
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
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame2, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
                # Determine the condition based on emergency detection
                condition = "Normal"
                if emergency_detected:
                    condition = "Emergency"
                    
                # Display the resulting frame
                cv2.imshow('jalan2', frame2)

            
        

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
        
        if kondisi == "Emergency":
            print("oke")
            
        elif kondisi == "Normal":
            if type(jalan1) == int and jalan1 > 0 and type(jalan4) == str:
                text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
                screen.blit(text_density, (10, 20))
                text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
                screen.blit(text_condition, (10, 60))
                text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
                screen.blit(text_vehicle_counter, (10, 100))
                text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
                screen.blit(text_green_light_duration, (10, 140))
                
                if not cek and type(jalan2) == str:
                    jalan2 = jalan1 + lampu_kuning
                    signal1 = greenSignal
                    cek = True    
                
                screen.blit(signal1, (360, 138))
                pygame.display.update()
                
                
                frame_counter += 1
                if frame_counter >= 8:
                    frame_counter = 0
                    jalan1 -= 1
                    text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                    screen.blit(text_jalan1, (330, 197))
                    
                    jalan2 -= 1
                    text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                    screen.blit(text_jalan2, (325, 430))                           

                if jalan1 == 0 and jalan2 == 3 and (ret1):
                    jalan1 = lampu_kuning +1
                    jalan1 -= 1
                    signal1 = yellowSignal
                    screen.blit(signal1, (360, 138))
                    wadah_jumlah = vehicle_counter
                        
                        
                if jalan1 == 0 and jalan2 == 0:
                    jumlah = wadah_jumlah
                    jalan1 = '---'
                    jalan2 = lampu_hijau
                    signal1 = redSignal
                    signal2 = greenSignal
                    screen.blit(redSignal, (360, 138))
                    screen.blit(signal2, (355, 418))
                    text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                    screen.blit(text_jalan1, (330, 197))
                    
            elif type(jalan2) == int and jalan2 > 0 and type(jalan1) == str:
                
                text_density = font.render(f'Kepadatan: {kepadatan}', True, (0, 0, 0))
                screen.blit(text_density, (10, 20))
                text_condition = font.render(f'Kondisi: {kondisi}', True, (0, 0, 0))
                screen.blit(text_condition, (10, 60))
                text_vehicle_counter = font.render(f'Kendaraan: {jumlah}', True, (0, 0, 0))
                screen.blit(text_vehicle_counter, (10, 100))
                text_green_light_duration = font.render(f'Durasi Lampu Hijau: {lampu_hijau}', True, (0, 0, 0))
                screen.blit(text_green_light_duration, (10, 140))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        
    # Release resources
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    pygame.quit()

if __name__ == '__main__':  
    # Path to the input videos
    video_path = ['C:/Users/Acer/yolov4/video/video1.mp4',
                  'C:/Users/Acer/yolov4/video/video2.mp4', 
                  'C:/Users/Acer/yolov4/video/video3.mp4',
                  'C:/Users/Acer/yolov4/video/video4.mp4',
                  'C:/Users/Acer/yolov4/video/video5.mp4']
    
    detect_objects()
    
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
    
    