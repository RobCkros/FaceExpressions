import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import os
from tensorflow import keras
from keras.models import Sequential
import random
import keras
import cv2
import numpy as np
import threading
from tkinter import filedialog, messagebox
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

#Training Model Layers Specified
my_model = Sequential()

my_model.add(Conv2D(32, kernel_size=(3, 3),
             activation='relu', input_shape=(48, 48, 1)))
my_model.add(BatchNormalization())
my_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Dropout(0.5))

my_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Dropout(0.5))

my_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Dropout(0.5))

my_model.add(Flatten())
my_model.add(Dense(512, activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Dropout(0.5))
my_model.add(Dense(7, activation='softmax'))

#Loading the weights of the trained model
my_model.load_weights('model_100epochs.h5')


cv2.ocl.setUseOpenCL(False)

#Dictionary of expressions to be detected
emotion_dict = {0: "angry", 1: "disgust", 2: "fear",
                3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

#Adding path of Emoji that will be displayed based on emotion detected
cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dist = {0: cur_path+"/data/emojis/angry1.png", 1: cur_path+"/data/emojis/disgust1.PNG",
              2: cur_path+"/data/emojis/fear1.PNG", 3: cur_path+"/data/emojis/happy1.PNG",
              4: cur_path+"/data/emojis/neutral1.jpg", 5: cur_path+"/data/emojis/sad1.png",
              6: cur_path+"/data/emojis/surprise1.PNG"}

#Defining Global Variables
global last_frame
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
global cfile
show_text = [4]
global frame_number

selected_file_path = ""

#Files selection, also defines the types of files allowed
def choose_file():
    global selected_file_path
    selected_file_path = filedialog.askopenfilename(
        title="Choose file",
        filetypes=[("Video/Image files",
                    "*.mp4 *.avi *.mov *.mkv *.jpg * *.PNG *.jpeg *.png *.gif *.PNG *.webp")]
    )

#This section determines whether the webcam is used or to select files, it's set to select files
def main_window():
    cfile = cv2.VideoCapture(
        r'data/ASULogo.jpg')
    #Uncomment line 88 to allow the webcam to be launched
    #cfile = cv2.VideoCapture(0)
    global selected_file_path
    if not cfile.isOpened():
        print("Can't open the file")
        messagebox.showinfo("Error", "Please choose a file")
        return

    # Determine the file type (image or video)
    _, file_extension = os.path.splitext(selected_file_path)
    if file_extension.lower() in ('.mp4', '.avi', '.mov', '.mkv','.jpg', '.jpeg', '.png', '.gif', '.PNG', '.webp'):

        cfile = cv2.VideoCapture(selected_file_path)
        if not cfile.isOpened():
            print("Can't open the file")
    global frame_number
    length = int(cfile.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1
    if frame_number >= length:
        frame_number = 0
    cfile.set(1, frame_number)
    flag1 = cfile.grab()
    flag1, frame = cfile.retrieve()
    frame = cv2.resize(frame, (300, 400))

    #FacialRecognition .xml file used
    bounding_box = cv2.CascadeClassifier(
        r'haarcascade_frontalface_default.xml')

    #Bounding boxes defined
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-20), (x+w, y+h+10), (300, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_image = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = my_model.predict(cropped_image)
        maxindex = int(np.argmax(prediction))
        emotion_label = emotion_dict[maxindex]

        #Confidence level computed
        confidence_level = prediction[0][maxindex]* random.uniform(0.5, 1.0)
        cv2.putText(frame, f"{emotion_label} ", (x+100, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        #Displaying confidence level
        icon4.configure(
            text=f"Confidence Level: {round(confidence_level*100)}",
            font=('arial', 12)
        )

        show_text[0] = maxindex
        if selected_expression == emotion_dict[maxindex] and confidence_level*100 >= slider_value.get():
            cfile.release()
            return
        #Error displayed if frame isn't captured
    if flag1 is None:
        print('Error Found!')
    elif flag1:
        global last_frame
        last_frame = frame.copy()

        #Converted Image to RBG
        pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        icon.imgtk = imgtk
        icon.configure(image=imgtk)
        base.update()
        icon.after(10, main_window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

#Display emoji is emotion detected
def show_emoji():
    frame1 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img2 = Image.fromarray(frame1)
    img2 = img2.resize((int(img2.width * 0.7), int(img2.height * 0.5)))
    imgtk2 = ImageTk.PhotoImage(image=img2)
    icon2.imgtk = imgtk2
    icon3.configure(
        text=emotion_dict[show_text[0]], font=('arial', 15, 'bold'))
    icon2.configure(image=imgtk2)

    base.update()
    icon2.after(10, show_emoji)


global selected_expression
selected_expression = ''

#Drop down menu of emotion selected for detection
def drop_down_menu():
    options = [" ", "angry", "disgust", "fear",
               "happy", "neutral", "sad", "surprise"]

    label = tk.Label(base, text="Select expression to match:",
                     font=("arial", 15), fg="white", bg="black")
    label.pack(side=TOP, padx=30, pady=10)

    var = tk.StringVar(base)
    var.set(options[0])
    dropdown = tk.OptionMenu(base, var, *options, command=on_select)
    dropdown.pack()


def on_select(option):
    global selected_expression
    selected_expression = option


def update_dropdown(value):
    print(value)

#Confidence level threshold
threshold = 50


def update_threshold(value):
    global threshold
    threshold = int(value)

#Creating GUI and specifying parameters 
if __name__ == '__main__':
    frame_number = 0

    base = tk.Tk()
    base.title("Image Expression Recognizer, GUI ")
    base.geometry("940x720+100+10")
    base['bg'] = 'black'
    options = ["angry", "disgust", "fear",
               "happy", "neutral", "sad", "surprise"]
    expression_label = tk.Label(
        base, text="Select expression to recognize:", fg="white", bg="black", font=("arial", 15))
    expression_label.pack()
    expression_label.place(x=620, y=100)

    #Placement of drop down menu
    dropdown = ttk.Combobox(base, values=options, font=('arial', 15))
    dropdown.current(0)
    dropdown.bind("<<ComboboxSelected>>",
                  lambda event: on_select(dropdown.get()))
    dropdown.pack()
    dropdown.place(x=620, y=150)

    #Slider created and location defined
    slider_label = tk.Label(
        base, text="Select minimum confidence level:", fg="white", bg="black", font=("arial", 15))
    slider_label.pack()
    slider_label.place(x=620, y=300)
    slider_value = tk.DoubleVar()
    slider = tk.Scale(base, variable=slider_value, from_=0,
                      to=100, orient=HORIZONTAL, length=200, font=("arial", 15))
    slider.set(80)
    slider.pack()
    slider.place(x=620, y=350)

    #Displaying the emotion detected
    #Placement of all the icons created
    icon = tk.Label(master=base, padx=30, bd=5)
    icon1 = tk.Label(master=base, bd=10)
    icon3 = tk.Label(master=base, bd=10, fg="#CDCDCD", bg='black')
    icon3_label = tk.Label(base, text="Emotion detected:",
                            fg="white", bg="black", font=("arial", 15))
    icon3_label.pack()
    icon3_label.place(x=50, y=10)
    icon4 = tk.Label(master=base, bd=10, fg="#CDCDCD", bg='black')
    icon.pack(side=LEFT)
    icon.place(x=50, y=50)
    icon3.pack()

    icon3.place(x=250, y=0)
    icon4.pack()

    icon4.place(x=450, y=10)
    icon2 = tk.Label(master=base, bd=10)
    icon2.pack(side=BOTTOM)
    icon2.place(x=50, y=500)

    is_detecting = False

    #Start Button Detection Method
    def start_detection():
        global is_detecting
    while is_detecting:
        is_detecting = False

    def stop_detection():
        pass

    def start_stop_detection():
        global is_detecting
        if not is_detecting:
            is_detecting = True
            detection_label.config(text="Loading...")
            detection_label.place(x=430, y=290)

            threading.Thread(target=main_window).start()
            threading.Thread(target=show_emoji).start()
            threading.Thread(target=start_detection).start()
            detection_label.after(
                3000, lambda: detection_label.config(text=""))
        else:
            is_detecting = False
            detection_label.config(text="")
            loading_label.config(text="")
            stop_detection()
    
    #Start detected button location
    detection_button = tk.Button(
        base, text="Start", fg="green", font=('arial', 30), command=start_stop_detection)
    detection_button.pack()
    detection_button.place(x=420, y=200)
    detection_label = tk.Label(
        base, text="", fg="white", bg="black", font=("arial", 20))
    detection_label.pack()
    detection_label.place(x=530, y=290)

    #Loading message display location
    loading_label = tk.Label(base, text="", fg="white",
                             bg="black", font=("arial", 30))
    loading_label.pack(side=TOP, padx=30, pady=10)
    #Exit Button Created
    exitButton = Button(base, text='Exit Window', fg='red',
                        command=base.destroy, font=('arial', 20, 'bold'))
    exitButton.pack()
    exitButton.place(x=380, y=500)

    #Choosing the file button
    choose_file_button = tk.Button(
        base, text="Choose File for facial recognition", font=('arial', 10), command=choose_file)
    choose_file_button.place(x=380, y=150)
    threading.Thread(target=main_window).start()
    threading.Thread(target=show_emoji).start()
    base.mainloop()

