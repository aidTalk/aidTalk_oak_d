#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:26:32 2021

@author: josesolla
"""

import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time

# Bring Gaze Tracker object!
from gaze_tracker import GazeTracker

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        
        # self.window_width = self.window.winfo_screenwidth()
        # self.window_height = self.window.winfo_screenheight()
        
        self.accumulated_time = 0
        self.contador = 0

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        
        ## Gaze tracker object
        self.gazeTracker = GazeTracker((int(self.vid.height), int(self.vid.width)), "photos/keyboard.jpg", self.vid.width, loadCalibration = False)


        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        ### Key pressed for calibration
        self.window.bind("<Key-c>",self.calibrateSystem)
        
        self.window.bind("<Key-s>",self.snapHeadPose)
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        
        self.window.mainloop()
        
    
    def calibrateSystem(self, event):
        self.gazeTracker.Calibrate()
    
    def snapHeadPose(self, event):
        self.gazeTracker.printInfo()

    def update(self):
        # Get a frame from the video source
        # start = time.time()
        ret, frame = self.vid.get_frame()
        
        
        if ret:
            # start = time.time()
            frame = cv2.flip(frame, 1)
            shown_img,_,_ = self.gazeTracker.Update(frame)
            shown_img = cv2.cvtColor(shown_img, cv2.COLOR_BGR2RGB)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(shown_img))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            # elapsed = time.time()-start
            # print(int(1/elapsed))
            
        self.window.after(self.delay, self.update)
        # self.contador = self.contador+1
        # self.accumulated_time =  self.accumulated_time + (time.time()-start) 
        # print("AVERAGE FPS: %0.3f" % (self.accumulated_time/self.contador))
        # print(self.accumulated_time)


class MyVideoCapture: 
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")