# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:56:21 2022

@author: Sahil
"""

import cv2
imageOpen = cv2.imread("image2.jpg") # read image
width = int(imageOpen.shape[1]*50/100) # width of image is reduced by scale factor 2
height = int(imageOpen.shape[0]*50/100) # height of image is reduced by scale factor 2
size1 = (width,height) 
imageResize = cv2.resize(imageOpen, size1) # cv2 method resize() resizes the original image
imageResize = cv2.blur(imageResize, (2, 2)) # blurs the image to try and reduce the background noise
imageGray = cv2.cvtColor(imageResize, cv2.COLOR_BGR2GRAY) # converting image to grayscale as thresholding only accepts grayscale images
ret, thresh = cv2.threshold(imageGray, 145, 255, cv2.THRESH_BINARY) # Binary thresholding, if pixel value is over 100, then set it to 255, otherwise 0
cv2.imshow("Original", imageResize) #showing original image for check
cv2.imshow("Grayscale", imageGray) # showing grayscale image
cv2.imshow("Binary", thresh) # showing results of Binary threshold
cv2.imwrite("denoised_image.jpg", thresh)
cv2.waitKey(1)

