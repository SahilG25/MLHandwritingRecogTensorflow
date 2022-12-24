# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:49:45 2022

@author: Sahil
"""
### python libraries
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
### my code libraries
import spacing_for_letter
import thickness_of_letter_enlarged
import TESTING_nn
import row_sorting
import laying_out_words
import detect_letters
###
nn = tf.keras.models.load_model("final_CNN") # loading my CNN model
name_of_file = "3_rows_roughly.png" # I will use this when I need to call functions from external python files

def open_file():
    global name_of_file
    global image
    global imageOpen
    filetypes = (
        ("PNG file", '*.png'),
        ("JPEG/JPG file", '*.jpeg *.jpg') # these are the only types of files acceptable when using filedialog.askopenfile() 
        )
    from tkinter import filedialog
    name_of_file = filedialog.askopenfilename( # this is a tkinter method that opens a window and allows the user to manually search their directories to open a file
        title = "Open a file",
        initialdir = "/",
        filetypes = filetypes # I have defined this above, PNG and JPEG/JPG are the only filetypes allowed
        )
    imageOpen = Image.open(name_of_file) # opening the file path that the user has input
    image = ImageTk.PhotoImage(image = imageOpen)
    canvas_image.itemconfig(image_plot, image = image) # displays the image onto canvas_image
    canvas_image.delete("rect")

root = Tk() #defining the main window of the program
menu1 = Menu(root) # the Menu() defines a menu for me in the window
menu_dropdown = Menu(menu1, tearoff = 0)
menu_dropdown.add_command(label = "Open", command = open_file) # this is the actual button that will allow the user to input images
menu_dropdown.add_separator()
menu1.add_cascade(label = "File operations", menu = menu_dropdown) # this is the tab the user has to click on to get to the "Open" button
root.config(menu = menu1) # stating that the main window will have a menubar
canvas_image = Canvas(root) # canvas widget is defined where I will place my input image
canvas_image.config(highlightthickness=0)
canvas_image.pack(side=LEFT, expand=YES, fill=BOTH) # .pack() allocates a coordinate space for canvas_image  
imageOpen = Image.open(name_of_file) # opening input image
image = ImageTk.PhotoImage(imageOpen) # PhotoImage object that will be used to output image in canvas_image
image_plot = canvas_image.create_image(0,0, anchor="nw",image=image) # .create_image() displayes my input image on the canvas
vertical_scrollbar = Scrollbar(root, orient = "vertical", command = canvas_image.yview) # defining vertical scrollbar
vertical_scrollbar.pack(side = RIGHT, fill = Y)
horizontal_scrollbar = Scrollbar(root, orient = "horizontal", command = canvas_image.xview) # defining horizontal scrollbar
horizontal_scrollbar.pack(side = BOTTOM, fill = X) # placement of this scrollbar is at the bottom
horizontal_scrollbar.config(command=canvas_image.xview)
vertical_scrollbar.config(command=canvas_image.yview)
canvas_image.configure(yscrollcommand = vertical_scrollbar.set, xscrollcommand = horizontal_scrollbar.set) # this is where I assign the scrollbar command onto canvas_image specifically
canvas_image.configure(scrollregion=canvas_image.bbox("all"))
canvas_image.config(scrollregion=(0,0,np.shape(imageOpen)[1], np.shape(imageOpen)[0]))
label = Label(canvas_image)
label.pack()
scale_factor = 1
def zoom(event):
    global image
    global scale_factor
    global counter
    scale_factor = slider1.get() # this .get() method receives the input that the user gives for Scale()
    width = int(np.shape(imageOpen)[1]*scale_factor/100) # I change the width of the original image using scale factor
    height = int(np.shape(imageOpen)[0]*scale_factor/100) # I change the height of the original image using scale factor
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    imageResize = imageOpen.resize((width, height)) # resizing image based on new width and height
    image = ImageTk.PhotoImage(image = imageResize)
    canvas_image.itemconfig(image_plot, image = image) # .itemconfig is used to update imageplot which deals with displaying images (in above code)
    canvas_image.delete("rect")
    resize_bounding_with_zoom()
    if counter == 1:
        recognition()
label = Label(root, text = "Zoom: ")
label.pack()
slider1 = Scale(root, from_ = 0, to = 200, orient = HORIZONTAL, command = zoom) # this is the slider that is used for zooming, with minimum value = 0, maximum value = 200
slider1.pack()
coor_info = []
def detect():
    global coor_info # I have used coor_info as a global variable so I can use it throughout the program
    import detect_letters # importing my program detect_letters
    coor_info = detect_letters.main(name_of_file) # coordinate information of letters
    height = np.shape(Image.open(name_of_file))[0]
    import row_sorting
    output = row_sorting.main(coor_info, height)
    #print("output: " + str(output))
    for x in range(0, len(coor_info)):
        x_coor = int(round(coor_info[x][0][0][0]*scale_factor/100)) # x coordinate of bounding box 
        y_coor = int(round(coor_info[x][0][0][1]*scale_factor/100)) # y coordinate of bounding box 
        x_coor_w = int(round(coor_info[x][0][1][0]*scale_factor/100)) # x + w coordinate of bounding box 
        y_coor_h = int(round(coor_info[x][0][1][1]*scale_factor/100)) # # y + h coordinate of bounding box
        #print(x_coor, y_coor, x_coor_w, y_coor_h)
        canvas_image.create_rectangle(x_coor, y_coor, x_coor_w, y_coor_h, outline = "blue", tags = "rect") # creates bounding box
        #text_to_print = "x:{}, y:{}, num: {}".format(x_coor_w, y_coor_h, coor_info[x][1]) # text for each bounding box showing its num (from detect letters) and current coordinates
        #canvas_image.create_text(x_coor_w, y_coor_h, text = text_to_print, fill = "black", tags = "rect") # outputs text
def resize_bounding_with_zoom():
    global coor_info # I have used coor_info from detect as a global variable so I can use it throughout the program
    global scale_factor
    for x in range(0, len(coor_info)):
        x_coor = int(round(coor_info[x][0][0][0]*scale_factor/100)) # x coordinate of bounding box after scale factor
        y_coor = int(round(coor_info[x][0][0][1]*scale_factor/100)) # y coordinate of bounding box after scale factor
        x_coor_w = int(round(coor_info[x][0][1][0]*scale_factor/100)) # x + w coordinate of bounding box after scale factor
        y_coor_h = int(round(coor_info[x][0][1][1]*scale_factor/100)) # # y + h coordinate of bounding box after scale factor
        #print(x_coor, y_coor, x_coor_w, y_coor_h)
        canvas_image.create_rectangle(x_coor, y_coor, x_coor_w, y_coor_h, outline = "blue", tags = "rect")
        #text_to_print = "x:{}, y:{}, num: {}".format(x_coor_w, y_coor_h, coor_info[x][1])
        #canvas_image.create_text(x_coor_w, y_coor_h, text = text_to_print, fill = "black",  tags = "rect")
detect_button = Button(root, text = "Detect", command = detect)
detect_button.pack()
def delete_canvas_rectangles():
    canvas_image.delete("rect")
delete_button = Button(root, text = "Delete", command = delete_canvas_rectangles)
delete_button.pack()
counter = 0
def recognition():
    global counter
    counter = 1
    global nn
    global scale_factor
    image_recog_list = []
    for x in range(0, len(coor_info)):
        num = coor_info[x][1] # this is the unique identifier for each letter (letter saved as "letter_image" + num)
        path = "letter_image{}.jpg".format(num) # path of letter
        #print("path of original: {}".format(path))
        
        img_letter = cv2.imread(path)  # image is loaded
        spacing_for_letter.main(img_letter, path) # spacing is applied on letter
        path = "spaces_{}".format(path)
        #print("path of spaces: {}".format(path))
        
        img_space = cv2.imread(path) # image is loaded
        thickness_of_letter_enlarged.main(img_space, path) # thickness of letter is enlarged
        path = "thick_{}".format(path)
        #print("path of thick: {}".format(path))
        
        image_for_pred = TESTING_nn.process_letter(path) # image is processed for tensorflow neural network
        output = nn.predict(image_for_pred) # the model.predict() takes input image and gives out an output class of size 47 containing probabilities of output
        result = np.argmax(output) # np.argmax() outputs the index of the highest probabilimty in the array
        alphabet = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']
        #alphabet is the mapping that maps the output index of result to a letter.
        image_recog_list.append(alphabet[result]) # the result of neural network (letter) is added to array
    for x in range(0, len(coor_info)):
        x_coor = int(round(coor_info[x][0][0][0]*scale_factor/100)) # x coordinate of bounding box after scale factor
        y_coor = int(round(coor_info[x][0][0][1]*scale_factor/100)) # y coordinate of bounding box after scale factor
        x_coor_w = int(round(coor_info[x][0][1][0]*scale_factor/100)) # x + w coordinate of bounding box after scale factor
        y_coor_h = int(round(coor_info[x][0][1][1]*scale_factor/100)) # # y + h coordinate of bounding box after scale factor
        #print(x_coor, y_coor, x_coor_w, y_coor_h)
        canvas_image.create_rectangle(x_coor, y_coor, x_coor_w, y_coor_h, outline = "blue", tags = "rect")
        #text_to_print = "{}".format(image_recog_list[x]) # result of recognition is shown
        canvas_image.create_text(x_coor_w, y_coor_h, text = text_to_print, fill = "black",  font=('Helvetica 15 bold') , tags = "rect")
def recognition_to_words():
    global name_of_file
    import detect_letters # importing my program detect_letters
    coor_info = detect_letters.main(name_of_file) # coordinate information of letters
    height = np.shape(Image.open(name_of_file))[0]
    width = np.shape(Image.open(name_of_file))[1]
    import row_sorting
    output_containing_rows = row_sorting.main(coor_info, height) # this output contains letters that are sorted out into rows
    import laying_out_words
    output_fully_structured = laying_out_words.main(output_containing_rows, width) # this output contains letters that are sorted out into rows and words in each line
    #print(output_fully_structured) # to visually observe the dimensions of output_fully_structured
    final_word_letter_list = []
    for x in range(0, len(output_fully_structured)): # x loops in output_fully_structured (number of rows)
        temp = []
        for y in range(0, len(output_fully_structured[x])): # y loops in output_fully_structured (inside each row)
           temp1 = []
           for z in range(0, len(output_fully_structured[x][y])): # z loops inside each word inside each row
               #print("x: {}, y: {}, z: {}".format(x, y, z))
               num = output_fully_structured[x][y][z][1] # this is the unique identifier for each letter (letter saved as "letter_image" + num)
               path = "letter_image{}.jpg".format(num) # path of letter
               #print("path of original: {}".format(path))
               img_letter = cv2.imread(path)  # image is loaded
               spacing_for_letter.main(img_letter, path) # spacing is applied on letter
               path = "spaces_{}".format(path)
               #print("path of spaces: {}".format(path))
               
               img_space = cv2.imread(path) # image is loaded
               thickness_of_letter_enlarged.main(img_space, path) # thickness of letter is enlarged
               path = "thick_{}".format(path)
               #print("path of thick: {}".format(path))
               
               image_for_pred = TESTING_nn.process_letter(path) # image is processed for tensorflow neural network
               output = nn.predict(image_for_pred) # the model.predict() takes input image and gives out an output class of size 47 containing probabilities of output
               result = np.argmax(output) # np.argmax() outputs the index of the highest probabilimty in the array
               alphabet = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']
               #alphabet is the mapping that maps the output index of result to a letter.
               #print("result of letter: {}".format(alphabet[result]))
               temp1.append(alphabet[result]) # the result of neural network (letter) is added to array
           temp.append(temp1)
        final_word_letter_list.append(temp)
    #print("final_words_letters: {}".format(final_word_letter_list))
    rows = []
    for x in range(0, len(final_word_letter_list)):
        in_row = ""
        for y in range(0, len(final_word_letter_list[x])):
            str1 = ""
            for z in range(0, len(final_word_letter_list[x][y])):
                str1 += final_word_letter_list[x][y][z]
            in_row += " " + str1
        rows.append(in_row)
    return rows
def new_window():
    global name_of_file
    new_root = Toplevel(root) # makes a new window
    label = Label(new_root, text = "Output of each row")
    label.pack()
    canvas_image2 = Canvas(new_root) # canvas widget is defined where I will place my input image
    coor_info = detect_letters.main(name_of_file) # coordinate information of letters
    height = np.shape(Image.open(name_of_file))[0]
    width = np.shape(Image.open(name_of_file))[1]
    coor_info = detect_letters.main(name_of_file)
    print("coor_info: {}".format(coor_info) )
    output_containing_rows = row_sorting.main(coor_info, height) # this output contains letters that are sorted out into rows
    print("output_containing_row: {}".format(output_containing_rows))
    row_sorted = []
    row = 0
    img_file = cv2.imread(name_of_file)
    string_words = recognition_to_words()
    for x in range(0, len(output_containing_rows)):
        row += 1
        row_sorted = laying_out_words.bubbleSort_sort_column(output_containing_rows[x])# within each row, the list is sorted by x coordinates
        temp = img_file[int(round(row_sorted[0][0][0][1])):int(round(row_sorted[-1][0][1][1])), int(round(row_sorted[0][0][0][0])): int(round(row_sorted[-1][0][1][0]))]
        cv2.imwrite("row{}.png".format(row), temp)
    save_ImageTk_instance = []
    save_pos_previous_image = 0
    entry_get = []
    for x in range(0, len(output_containing_rows)):
        save_pos_previous_image += 70
        #print("in here")
        filename = "row{}.png".format(x+1)
        file_open = Image.open(filename)
        file_open = file_open.resize((700,60), Image.ANTIALIAS)
        a = Entry(canvas_image2, width = 60)
        a.insert(0, string_words[x])
        canvas_image2.create_window(0, save_pos_previous_image + 35 , anchor = "w", window = a)
        entry_get.append(a)
        photo = ImageTk.PhotoImage(file_open)
        save_ImageTk_instance.append(photo)
        canvas_image2.create_image(0,save_pos_previous_image,anchor="w", image = save_ImageTk_instance[-1])
    def writeDoc():
        #global entry_get
        final_write_list = []
        openFile = open("generated_output.txt", "w")
        for x in range(0, len(entry_get)):
            openFile.write("\n" + entry_get[x].get())
        openFile.close()
        from tkinter import messagebox
        messagebox.showinfo("showinfo", "File saved as generated_output.txt")
    vertical_scrollbar = Scrollbar(new_root, orient = "vertical", command = canvas_image.yview) # defining vertical scrollbar
    vertical_scrollbar.pack(side = RIGHT, fill = Y)
    horizontal_scrollbar = Scrollbar(new_root, orient = "horizontal", command = canvas_image.xview) # defining horizontal scrollbar
    horizontal_scrollbar.pack(side = BOTTOM, fill = X) # placement of this scrollbar is at the bottom
    horizontal_scrollbar.config(command=canvas_image2.xview)
    vertical_scrollbar.config(command=canvas_image2.yview)
    canvas_image2.configure(yscrollcommand = vertical_scrollbar.set, xscrollcommand = horizontal_scrollbar.set) # this is where I assign the scrollbar command onto canvas_image specifically
    canvas_image2.configure(scrollregion=canvas_image2.bbox("all"))
    canvas_image2.config(highlightthickness=0)
    canvas_image2.pack(side=LEFT, expand=YES, fill=BOTH) # .pack() allocates a coordinate space for canvas_image
    generate_output = Button(new_root, text = "Generate output", command = writeDoc)
    generate_output.pack()
    new_root.mainloop()
new_window_button = Button(root, text = "Window to display output" , command = new_window)
new_window_button.pack()
root.mainloop()






        