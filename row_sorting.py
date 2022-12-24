# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:44:52 2022

@author: Sahil
"""
def bubbleSort_sort_row(list1): # I will perform bubble sort on my array
    while True: # while loop doesn't stop until there is no adjacency swap
        swap = False
        for x in range(0, len(list1)-1):
            if list1[x][0][1][1] > list1[x+1][0][1][1]: # if the adjacent (to the right) element is smaller than the current then swap
                swap = True 
                temp = list1[x]
                list1[x] = list1[x+1]
                list1[x+1] = temp # elements swapped[4]: row_sorted = main(coor_info)
        if swap == False:
            return list1 # if there is no swaps in the list anymore, then bubble sort is complete, return new list
def main(coor_info, height): #I have made a function that takes the list of coordinates from detect_letter.py as a parameter
    y_sorted = bubbleSort_sort_row(coor_info) # first we need to bubble sort the coordinates so adjacent elements can be compared
    # note the relative position inside a line doesn't matter yet (we will rearrange that once we work out our arrays of each line)
    rows_sorted = []
    temp = 0 # I will use temp in slicing of my array to mark the first element of a new line
    for x in range(0, len(y_sorted)-1):
        if abs(y_sorted[x][0][1][1] - y_sorted[x+1][0][1][1])/height > 0.07: # if the difference between two y-coordinates (adjacent)/height > 0.07 means there is a new line 
            rows_sorted.append(y_sorted[temp: x + 1]) # a new row is added with slicing between first index of the previous line (temp) and current index
            temp = x+1 # the current index becomes the new temp
        if x == len(y_sorted)-2: # I noticed during testing that the last line of the image was not saved, so this is an if statement to fix that
            rows_sorted.append(y_sorted[temp: x+2])
    return rows_sorted # return sorted list


