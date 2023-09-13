# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:57:58 2023

@author: Acer
"""

#generate file.txt database
import glob

def generate_txt(root_dir, filename):
  """ 
  Generate a txt file containing the path to the images in the `root_dir` directory
  """
  images_list = glob.glob(root_dir + "*.jpg")

  # Replace backslashes with forward slashes
  images_list = [path.replace("\\", "/") for path in images_list]

  with open(filename, "w") as f:
    f.write("\n".join(images_list))
  
generate_txt("/Users/Acer/yolov4/train/", "train.txt")
generate_txt("/Users/Acer/yolov4/valid/", "valid.txt")
generate_txt("/Users/Acer/yolov4/test/", "test.txt")