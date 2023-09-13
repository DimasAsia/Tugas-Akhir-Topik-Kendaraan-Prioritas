import cv2
import numpy as np
from object_deteksi import detect_objects 


# Load video
video = cv2.VideoCapture("/Users/Acer/yolov4/test2.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect objects in the frame
    frame = detect_objects(frame)
    resized_frame = cv2.resize(frame, (608, 608))

    # Display the resulting frame
    cv2.imshow("Object Detection", resized_frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
