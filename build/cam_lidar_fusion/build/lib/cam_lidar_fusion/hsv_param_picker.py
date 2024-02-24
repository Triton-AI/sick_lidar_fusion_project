import cv2
import numpy as np
import tkinter as tk


# Initialize the video stream using the VideoCapture function
cap = cv2.VideoCapture(0)


def update_frame(Hl, Hh, Sl, Sh, Vl, Vh):
    ret, frame = cap.read()
    blur = cv2.GaussianBlur(frame, (5, 5), 3)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_range = np.array([Hl, Sl, Vl], dtype=np.uint8)
    upper_range = np.array([Hh, Sh, Vh], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('Result', result)

root = tk.Tk()
root.geometry('500x500')

# Create a frame to contain the sliders
slider_GUI = tk.Frame(root)
slider_GUI.pack()

# Create the first slider
low_H = tk.Scale(slider_GUI, label='low H', from_=0, to=180, orient='horizontal')
low_H.set(0)
low_H.pack()

# Create the second slider
high_H = tk.Scale(slider_GUI, label='high H', from_=0, to=180, orient='horizontal')
high_H.set(180)
high_H.pack()

# Create the third slider
low_S = tk.Scale(slider_GUI, label='low S', from_=0, to=255, orient='horizontal')
low_S.set(0)
low_S.pack()

# Create the fourth slider
high_S = tk.Scale(slider_GUI, label='high S', from_=0, to=255, orient='horizontal')
high_S.set(255)
high_S.pack()

# Create the second slider
low_V = tk.Scale(slider_GUI, label='low V', from_=0, to=255, orient='horizontal')
low_V.set(0)
low_V.pack()

# Create the third slider
high_V = tk.Scale(slider_GUI, label='high V', from_=0, to=255, orient='horizontal')
high_V.set(255)
high_V.pack()

# Set the command option for each slider to a lambda function that calls update_frame
low_H.config(command=lambda value: update_frame(value, high_H.get(), low_S.get(), high_S.get(), low_V.get(), high_V.get()))
high_H.config(command=lambda value: update_frame(low_H.get(), value, low_S.get(), high_S.get(), low_V.get(), high_V.get()))
low_S.config(command=lambda value: update_frame(low_H.get(), high_H.get(), value, high_S.get(), low_V.get(), high_V.get()))
high_S.config(command=lambda value: update_frame(low_H.get(), high_H.get(), low_S.get(), value, low_V.get(), high_V.get()))
low_V.config(command=lambda value: update_frame(low_H.get(), high_H.get(), low_S.get(), high_S.get(), value, high_V.get()))
high_V.config(command=lambda value: update_frame(low_H.get(), high_H.get(), low_S.get(), high_S.get(), low_V.get(), value))


# Start the main loop to continuously update the video stream
# while True:
#     update_frame(low_H.get(), high_H.get(), low_S.get(), high_S.get(), low_V.get(), high_V.get())
#     key = cv2.waitKey(1)
#     if key == 27:  # Press Escape to exit
#         break
#
# cap.release()
# cv2.destroyAllWindows()

root.mainloop()