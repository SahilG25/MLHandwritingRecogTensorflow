# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 00:45:23 2022

@author: Sahil
"""

import numpy as np
import cv2
def main(x, path): # since this is meant to be used for multiple images, I will make a subroutine instead of a sequence, so multiple the function can be called several times for use instead of repeating a sequence (unnessary lines of code)
    width = x.shape[1] # since x is cv2.imread() array, I can use .shape() method to retrieve width and height of the image
    height = x.shape[0]
    temp_counter = False # this is a boolean variable that will me changed to True if any of the dimension parameters of x are odd
    if int(x.shape[0])%2 != 0:
        temp_counter = True
        height += 1 # if height is odd, increment it to make it even
    if int(x.shape[1])%2 != 0:
        temp_counter = True
        width += 1 # if width is odd, increment it to make it even
    if temp_counter == True:
        size1 = (width, height) # if temp_counter = True, then change the dimensions of x using cv2.resize()
        x = cv2.resize(x, size1)
    
    blank_image = np.full((height*2, width*2, 3), 255, dtype = np.uint8) # np.full() creates a numpy array with identical values 255 (thats the value for a plain white image) in the dimensions of x multiplied by a scale factor 2
    blank_image[int(blank_image.shape[0]/2) - int(round(x.shape[0]/2)):int(blank_image.shape[0]/2) + int(round(x.shape[0]/2)) , int(blank_image.shape[1]/2) - int(round(x.shape[1]/2)):int(blank_image.shape[1]/2) + int(round(x.shape[1]/2))] = x
    # since we need x to be in the centre of blank_image, we need to substitute x between x = centre of blank_image(x coordinate) + half of x(its x coordinate) and centre of blank_image(x coordinate) - half of x (its x coordinate), and y = centre of blank_image(y coordinate) + half of x(its y coordinate) and centre of blank_image(y coordinate) - half of x (its y coordinate)
    cv2.imwrite("spaces_{}".format(path), blank_image)
#a = cv2.imread("letter_image30.jpg")
#main(a, "letter_image30.jpg")
#cv2.waitKey(0)
