# -*- coding: utf-8 -*-
"""
Created on Mon May  1 22:33:25 2023

@author: Acer
"""

# Python code equivalent of the given bash commands
import os

# Append class names to 'classes.names' file
with open('classes.names', 'a') as f:
    f.write("ambulance\n")
    f.write("mobil\n")
    f.write("motor\n")
    f.write("pemadam\n")
    f.write("truk\n")

# Write configuration to 'darknet.data' file
with open('darknet.data', 'w') as f:
    f.write("classes= 5\n")
    f.write("train  = /Users/Acer/yolov4/train.txt\n")
    f.write("valid  = /Users/Acer/yolov4/valid.txt\n")
    f.write("names = /Users/Acer/yolov4/classes.names\n")
    f.write("backup = /Users/Acer/yolov4")
