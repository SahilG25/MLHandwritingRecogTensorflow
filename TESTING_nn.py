# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:54:44 2022

@author: Sahil
"""
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
#nn = tf.keras.models.load_model("final_CNN") # loading my CNN model
#from PIL import Image, ImageOps
def process_letter(img_path):
    image = Image.open(img_path) # opening an image using its path
    image_resized = image.resize((28,28)) # resizing image to 28 by 28 so that inverse and grayscale operations can be performed
    image_gray = image_resized.convert("L") # converts to grayscale
    image_inverse = ImageOps.invert(image_gray) # inverts the image so the background is black and foreground is white
    numpy_image = np.array(image_inverse) # converting to numpy array for reshaping it for CNN
    final1 = numpy_image.reshape(1, 28, 28, 1) # resizing array for CNN
    final1 = final1/255 # normalising the data so it is continuous
    return final1

'''def main(path):
    image_for_pred = process_letter(path)
    output = nn.predict(image_for_pred) # the model.predict() takes input image and gives out an output class of size 47 containing probabilities of output
    result = np.argmax(output) # np.argmax() outputs the index of the highest probabilimty in the array
    alphabet = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']
    #alphabet is the mapping that maps the output index of result to a letter.
    return alphabet[result]'''
"""while True: # user interface
    input1 = input("Enter image path: ") # image path of letter that needs to be recognised
    from pathlib import Path
    if Path(input1).is_file() == True:
        if input1[len(input1)-4:] == ".jpg" or input1[len(input1)-4:] == ".png" or input1[len(input1)-4:] == ".jpeg":
            #image_for_pred = prepare_for_processing(input1) # image prepared for cnn
            image_for_pred = process_letter(input1)
            output = nn.predict(image_for_pred) # the model.predict() takes input image and gives out an output class of size 47 containing probabilities of output
            result = np.argmax(output) # np.argmax() outputs the index of the highest probabilimty in the array
            alphabet = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']
            #alphabet is the mapping that maps the output index of result to a letter.
            print("Your output is: {}".format(alphabet[result]))
        else:
            print("Sorry, must be .PNG/.JPG/.JPEG")
    else:
        print("Sorry, try again, not a valid path")"""
