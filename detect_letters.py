# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:28:20 2022

@author: Sahil
"""

import cv2
import numpy as np
def main(image1):
    imageOpen = cv2.imread(image1) # read image
    width = int(imageOpen.shape[1]*200/100) # width of image is reduced by scale factor 2
    height = int(imageOpen.shape[0]*200/100) # height of image is reduced by scale factor 2
    size1 = (width,height) 
    imageResize = cv2.resize(imageOpen, size1) # cv2 method resize() resizes the original image
    '''imageResize = cv2.blur(imageResize, (2, 2)) # blurs the image to try and reduce the background noise
    imageGray = cv2.cvtColor(imageResize, cv2.COLOR_BGR2GRAY) # converting image to grayscale as thresholding only accepts grayscale images
    ret, thresh = cv2.threshold(imageGray, 150, 255, cv2.THRESH_BINARY) # Binary thresholding, if pixel value is over 100, then set it to 255, otherwise 0
    contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # this is the method of cv2 that finds contours in an image (i.e. edges)
    # the array contours contains coordinates (x,y) of each contour detected (where it starts and ends) in the image
    '''
    grayScale = cv2.cvtColor(imageResize, cv2.COLOR_BGR2GRAY)
    unwanted, binary = cv2.threshold(grayScale, 200, 255, cv2.THRESH_BINARY)
    unwanted1, invBinary = cv2.threshold(binary, 50, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(invBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num = 0
    coor_info = []
    for c in range(0, len(contours)): # I can use contour coordinates to locate letters, I will therefore use a loop to print each contour detected in the image
        x, y, w, h = cv2.boundingRect(contours[c]) # this cv2.boundingRect() outputs the x,y coordinates and the width and height of each contour, I can use this information to plot rectangles on the image where contours are.
        area_of_contour = cv2.contourArea(contours[c]) # cv2.contourArea outputs the area of the detected contour, I could have made this function myself but this method is more mathematical and gives an accurate measure of area.
        if round(area_of_contour) < 250.0: # if the area of contour is less than a certain threshold, I will count the counter as noise and therefore ignore it (next iteration of contours)
            continue
        num += 1
        crop = imageResize[y:y+h, x:x+w] # I use numpy slicing to extract the region of the image containing the contour
        name = "letter_image{}.jpg".format(num)
        cv2.imwrite(name, crop) # saving each contour
        #rect = cv2.rectangle(imageResize, (x, y), (x + w, y + h), (0, 255, 255)) # plotting a rectangle where the contour is, using outputs of cv2.BoundingRect()
        decreasing_fac = 2
        coor_info.append([[(x/decreasing_fac,y/decreasing_fac), ((x+w)/decreasing_fac, (y+h)/decreasing_fac)], num])
    #cv2.imshow("a", imageResize)
    #cv2.waitKey()
    return coor_info

