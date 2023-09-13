# -*- coding: utf-8 -*-
"""
Created on Sun May 21 13:12:33 2023

@author: Acer
"""
#import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os, sys
import glob
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tkinter import *
from tkinter import messagebox

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.6, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

#load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov4.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4_1.cfg"])

# Create the interface object
trafficVisualization = Tk()
trafficVisualization.title("Countdown Timer")
trafficVisualization.geometry("1280x720")
trafficVisualization.configure(background='orange')

# Image Background Setting
img = PhotoImage(file="guibg.png")
bg = Label(trafficVisualization, image=img)
bg.place(x=0, y=0)

# TL Circle Set
imgGreenLight = PhotoImage(file="greenlight.png")
imgYellowLight = PhotoImage(file="yellowlight.png")
imgRedLight = PhotoImage(file="redlight.png")

green = Label(trafficVisualization, image=imgGreenLight)
yellow = Label(trafficVisualization, image=imgYellowLight)
red = Label(trafficVisualization, image=imgRedLight)

# Declare variables
secondStringA = StringVar()
secondStringB = StringVar()
secondStringC = StringVar()

mobilstring = StringVar()
motorstring = StringVar()

def createVisual(motor, mobil, lampu_hijau, lampu_merah):
    # Set strings to default value
    secondStringA.set(str(lampu_hijau))
    secondStringC.set("05")
    secondStringB.set(str(lampu_merah))
    mobilstring.set(str(mobil))
    motorstring.set(str(motor))

    # Get user input
    secondTextboxA = Entry(trafficVisualization, width=3, font=("Helvetica Neue", 12))
    secondTextboxB = Entry(trafficVisualization, width=3, font=("Helvetica Neue", 12))
    mobiltextbox = Entry(trafficVisualization, width=3, font=("Helvetica Neue", 12))
    motortextbox = Entry(trafficVisualization, width=3, font=("Helvetica Neue", 12))

    # Center textboxes
    secondTextboxA.place(x=420, y=410)
    secondTextboxB.place(x=620, y=280)
    mobiltextbox.place(x=300, y=35)
    motortextbox.place(x=300, y=85)

def runTimerB():
    try:
        clockTime2 = int(secondStringA.get()) + int(secondStringC.get())
        clockTime = int(secondStringB.get())
    except:
        print("Incorrect values")
    green_time = int(secondStringA.get())
    yellow_time = int(secondStringC.get())
    
    while clockTime > -1 and clockTime2 > -1:
        secondStringA.set("{0:02d}".format(clockTime2))
        secondStringB.set("{0:02d}".format(clockTime))
        
        if green_time > -1:
            green.place(x=735, y=285)
            red.place(x=515, y=415)
            yellow.place_forget()
            green_time = 1
        else:
            yellow.place(x=720, y=285)
            red.place(x=515, y=415)
            green.place_forget()
        
        trafficVisualization.update()
        time.sleep(1)
        
        if clockTime == 0 and clockTime2 == 0:
            break
        
        clockTime -= 1
        clockTime2 -= 1

def runTimerA():
    try:
        clockTime = int(secondStringA.get()) + int(secondStringC.get())
        clockTime2 = int(secondStringB.get())
    except:
        print("Incorrect values")
    green_time = int(secondStringA.get())
    yellow_time = int(secondStringC.get())
    
    while clockTime > -1 and clockTime2 > -1:
        secondStringA.set("{0:02d}".format(clockTime))
        secondStringB.set("{0:02d}".format(clockTime2))
        
        if green_time > -1:
            green.place(x=475, y=415)
            red.place(x=705, y=285)
            yellow.place_forget()
            green_time -= 1
        else:
            yellow.place(x=500, y=415)
            green.place_forget()
            red.place(x=705, y=285)
        
        trafficVisualization.update()
        time.sleep(1)
        
        if clockTime == 0 and clockTime2 == 0:
            break
        
        clockTime -= 1
        clockTime2 -= 1


def createJOI(motor, mobil, lampu_merah, lampu_hijau):
    img = np.zeros([512, 512, 3], dtype=np.uint8)
    img.fill(255)
    cv2.line(img, (170, 0), (170, 190), (0, 0, 0), 2)
    cv2.line(img, (320, 0), (320, 190), (0, 0, 0), 2)
    cv2.line(img, (170, 340), (170, 512), (0, 0, 0), 2)
    cv2.line(img, (320, 340), (320, 512), (0, 0, 0), 2)
    cv2.line(img, (0, 190), (170, 190), (0, 0, 0), 2)
    cv2.line(img, (0, 340), (170, 340), (0, 0, 0), 2)
    cv2.line(img, (320, 190), (512, 190), (0, 0, 0), 2)
    cv2.line(img, (320, 340), (512, 340), (0, 0, 0), 2)

    cv2.putText(img, f'Jumlah Mobil: {mobil}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'Jumlah Motor: {motor}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'Durasi Lampu Hijau: {lampu_hijau}', (330, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'Durasi Lampu Merah: {lampu_merah}', (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Simulations", img)
    cv2.waitKey(0)

def countdown(t):
    while t:
        mins, secs = divmod(t, 60) 
        timer = '(:02d):(:02d}'.format(mins, secs)
        print(timer, end-"\r")
        time.sleep(1)
        t -= 1

def detect_objects(outputs, frame):
    boxes = []
    confidences = []
    classIDs = []
    mobil = 0
    motor = 0
    (H, W) = frame.shape[:2]
    
    for output in outputs: 
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == 2 or classID == 3 or classID == 5 or classID == 7:
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            if classIDs[i] == 3:
                motor += 1
            else:
                mobil += 1
                
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return motor, mobil


def fuzzyrules(mobil, motor):
    number_of_car = ctrl.Antecedent(np.arange(0, 11, 1), 'number_of_car') 
    number_of_motorcycle = ctrl.Antecedent(np.arange(0, 21, 1), 'number_of_motorcycle') 
    traffic_light_duration = ctrl.Consequent(np.arange(0, 81, 1), 'traffic_light_duration')

    number_of_car['sepi'] = fuzz.trimf(number_of_car.universe, [0, 0, 4]) 
    number_of_car['sedang'] = fuzz.trimf(number_of_car.universe, [2, 4, 8]) 
    number_of_car['ramai'] = fuzz.trimf(number_of_car.universe, [4, 8, 11])
    
    number_of_motorcycle['sepi'] = fuzz.trimf(number_of_motorcycle.universe, [0, 0, 5]) 
    number_of_motorcycle['sedang'] = fuzz.trimf(number_of_motorcycle.universe, [5, 10, 15]) 
    number_of_motorcycle['ramai'] = fuzz.trimf(number_of_motorcycle.universe, [10, 20, 20])
    
    traffic_light_duration['short'] = fuzz.trimf(traffic_light_duration.universe, [0, 0, 20])
    traffic_light_duration['average'] = fuzz.trimf(traffic_light_duration.universe, [10, 40, 70])
    traffic_light_duration['long'] = fuzz.trimf(traffic_light_duration.universe, [60, 80, 80])
    
    rule1 = ctrl.Rule(number_of_car['sepi'] & number_of_motorcycle['sepi'], traffic_light_duration['short'])
    rule2 = ctrl.Rule(number_of_car['sedang'] & number_of_motorcycle['sedang'], traffic_light_duration['average'])
    rule3 = ctrl.Rule(number_of_car['ramai'] & number_of_motorcycle['ramai'], traffic_light_duration['long'])
    rule4 = ctrl.Rule(number_of_car['sepi'] & number_of_motorcycle['sedang'], traffic_light_duration['short'])
    rule5 = ctrl.Rule(number_of_car['sepi'] & number_of_motorcycle['ramai'], traffic_light_duration['average'])
    rule6 = ctrl.Rule(number_of_car['sedang'] & number_of_motorcycle['sepi'], traffic_light_duration['average'])
    rule7 = ctrl.Rule(number_of_car['sedang'] & number_of_motorcycle['ramai'], traffic_light_duration['long'])
    rule8 = ctrl.Rule(number_of_car['ramai'] & number_of_motorcycle['sepi'], traffic_light_duration['average'])
    rule9 = ctrl.Rule(number_of_car['ramai'] & number_of_motorcycle['sedang'], traffic_light_duration['long'])
    
    traffic_light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    traffic_light = ctrl.ControlSystemSimulation(traffic_light_ctrl)

    traffic_light.input['number_of_car'] = mobil
    traffic_light.input['number_of_motorcycle'] = motor
    traffic_light.compute()
    lampu_hijau = round(traffic_light.output['traffic_light_duration'])
    lampu_merah = lampu_hijau + 5
    
    return lampu_hijau, lampu_merah
        

#load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] Loading YOLO from disk...")
net cv2.dnn.readNet FromDarknet (configPath, weightsPath)
imgnames glob.glob("captured/*.jpg") 
for ingname in ingnames:
    frame = cv2.imread(ingname)
#determine only the output layer names that we need from YOLO 
    ln net.getLayerNames()
    ln [In[i[0] 1] for i in net.getUnconnectedOutLayers()] 
    #construct a blob from the input image and then perform a forward 
    #pass of the YOLO object detector, giving us our bounding boxes and 
    # associated probabilities

    blob = cv2.dnn.blobFromImage(frame, 1/ 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start time.time()
    layerOutputs net.forward(In) 
    end time.time()
    #show timing information on YOLO
    print("[INFO] YOLO took (:.6f) seconds".format(end - start))
    motor, mobil findObjects (layerOutputs, frame) lampu hijau, lampu merah fuzzyrules (mobil, motor) createVisual (motor, mobil,lampu_hijau, lampu_merah)
    print(ingname)
    if imgnames.index(imgname) % 2 == 0: 
        runTimerA()
    else:
        runTimerB()