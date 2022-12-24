# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 02:12:34 2022

@author: Sahil
"""
import cv2
def main(x, path): # since this is meant to be used for multiple images, I will make a subroutine instead of a sequence, so multiple the function can be called several times for use instead of repeating a sequence (unnessary lines of code)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    gray1 = gray.copy()
    et, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gray, contours,-1 ,(0, 0, 0), 2,cv2.LINE_AA) # this method of cv2 draws out the contours (output of cv2.findContours()) onto the original
    cv2.rectangle(gray,(0,0),(gray1.shape[1], gray1.shape[0]), (255,255,255),13)
    cv2.imwrite("thick_{}".format(path), gray)
#a = cv2.imread("spacing_letter_image14.jpg")
#main(a, "spacing_letter_image14.jpg")

