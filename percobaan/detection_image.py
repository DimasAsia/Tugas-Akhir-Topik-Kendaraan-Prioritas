import os
import cv2
import numpy as np
import random
from object_deteksi import detect_objects

# Set the path to the folder containing the images
path_to_images = 'C:/Users/Acer/yolov4/test/image2/ambulance2_jpg.rf.c30b81c86054b650b3697ea61b964de6.jpg'

# Load class names
classes = []
with open('/Users/Acer/yolov4/classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Select a random image from the folder
#random_image_path = os.path.join(path_to_images, random.choice(os.listdir(path_to_images)))

# Load image
image = cv2.imread(path_to_images)

# Detect objects in the image using the detect_objects function
result = detect_objects(image)

# Display the resulting image
cv2.imshow('Detected Objects', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
