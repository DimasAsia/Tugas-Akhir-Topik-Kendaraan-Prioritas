# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:39:10 2023

@author: Acer
"""
import cv2
import numpy as np
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def createJOI(density, condition, vehicle_counter, green_light_duration):
    img = np.zeros([512, 512, 3], dtype=np.uint8)
    img.fill(255)
    
    # Garis pada setiap sisi
    cv2.line(img, (0, 190), (170, 190), (0, 0, 0), 2)
    cv2.line(img, (0, 190), (0, 340), (0, 0, 0), 2)
    cv2.line(img, (0, 340), (170, 340), (0, 0, 0), 2)
    cv2.line(img, (170, 0), (170, 190), (0, 0, 0), 2)
    cv2.line(img, (170, 340), (170, 512), (0, 0, 0), 2)
    cv2.line(img, (320, 0), (320, 190), (0, 0, 0), 2)
    cv2.line(img, (320, 190), (512, 190), (0, 0, 0), 2)
    cv2.line(img, (320, 340), (512, 340), (0, 0, 0), 2)
    cv2.line(img, (320, 340), (320, 512), (0, 0, 0), 2)
    
    # Teks informasi
    cv2.putText(img, f'Kepadatan: {density}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'Kondisi: {condition}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'Kendaraan: {vehicle_counter}', (330, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'Durasi Lampu Hijau: {green_light_duration}', (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Kotak counter angka descending pada 4 siku garis tengah
    cv2.rectangle(img, (130, 150), (170, 190), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, '15', (130, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(img, (130, 340), (170, 380), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, '14', (130, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(img, (320, 150), (360, 190), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, '13', (320, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(img, (320, 340), (360, 380), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, '12', (320, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Simulations", img)
    cv2.waitKey(0)

density = 10
vehicle_counter = 10
green_light_duration = 1
condition = 0


createJOI(density, condition, vehicle_counter, green_light_duration)