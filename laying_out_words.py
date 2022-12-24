# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:11:58 2022

@author: Sahil
"""

def bubbleSort_sort_column(list1): # I will perform bubble sort on my array
    while True: # while loop doesn't stop until there is no adjacency swap
        swap = False
        for x in range(0, len(list1)-1):
            if list1[x][0][1][0] > list1[x+1][0][1][0]: # if the adjacent (to the right) x coordinate is smaller than the current then swap
                swap = True 
                temp = list1[x]
                list1[x] = list1[x+1]
                list1[x+1] = temp # elements swapped
        if swap == False:
            return list1 # if there is no swaps in the list anymore, then bubble sort is complete, return new list
def main(sorted_rows, width): #I have made a function that takes the list of coordinates from row_sorting.py as a parameter
    new_sorted_individual_row = []
    for x in range(0, len(sorted_rows)):
        a = bubbleSort_sort_column(sorted_rows[x]) # within each row, the list is sorted by x coordinates
        new_sorted_individual_row.append(a) # new list is appended to new_sorted_individual_row
    #return new_sorted_individual_row
    words_list_line_by_line = [] # this will store the rows and the word in the image
    for x in range(0, len(new_sorted_individual_row)):
        temp = 0  # I will use temp in slicing of my array to mark the first element of a new word
        word_row_list = [] # this stores the words made within each row
        for y in range(0, len(new_sorted_individual_row[x])-1):
            print("num inside loop: {}".format(new_sorted_individual_row[x][y][1]))
            if abs(new_sorted_individual_row[x][y][0][1][0] - new_sorted_individual_row[x][y+1][0][1][0])/width > 0.08: # if the difference between two adjacent x-coordinates/width > 0.095, form a word
                print("entered inside if statement at x = {}, y = {}, num = {}".format(x,y,new_sorted_individual_row[x][y][1]))    
                #print("slicing {}".format(new_sorted_individual_row[x][temp:y+1]))
                word_row_list.append(new_sorted_individual_row[x][temp:y+1]) # add the elements between temp and current index to word_row_list (new word)
                temp = y+1 # update temp to index of first letter in a new word 
            if y == len(new_sorted_individual_row[x])-2: # I noticed during testing that the last line of the image was not saved, so this is an if statement to fix that
                word_row_list.append(new_sorted_individual_row[x][temp: y+2])
        words_list_line_by_line.append(word_row_list) # add each row of words to words_list_line_by_line
    return words_list_line_by_line # output

