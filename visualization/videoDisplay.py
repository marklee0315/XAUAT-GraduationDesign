from tkinter import *
from tkinter import filedialog

import cv2
from PIL import ImageTk, Image

root = Tk()
root.title('关节信息行为识别')
root.geometry('800x500')

# Create a frame
mainFrame = Frame(root, bg="white")
mainFrame.grid()
# Create a label in the frame
mainLabel = Label(mainFrame)
mainLabel.grid()

# Capture from path
videoPath = filedialog.askopenfilename()
cap = cv2.VideoCapture(videoPath)


# function for video streaming
def videoStream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    mainLabel.imgtk = imgtk
    mainLabel.configure(image=imgtk)
    mainLabel.after(1, videoStream)


videoStream()
root.mainloop()


def executeFrom():
    print('1')
