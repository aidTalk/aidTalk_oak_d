#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:22:20 2021

@author: josesolla
"""

import cv2
from gaze_tracker import GazeTracker
import time
import tkinter as tk


###################################################

# ----- Program Functions actioned by keys: ----- #

    # To Calibrate, press 'c'
    # To Activate the Blinking Counter, press 'b'
    # To Exit the program, press 'q'

###################################################



# Capture Initialization
#print("antes de videocapture")
cap = cv2.VideoCapture(0)

# Window setting
#print("antes de window")
windowName = "aidTalk"
#cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)


# Create GazeTracker object
### Get screen resolution
#print("antes de tk")
root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.destroy()
del root

gazeTracker = GazeTracker(screen_size=(width, height), camera_resolution=(1280, 720))

# alpha_slider_max = 100
# def on_trackbar(val):
#     print(val)

# trackbar_name = 'Alpha x %d' % alpha_slider_max
# cv2.createTrackbar(trackbar_name, windowName , 0, alpha_slider_max, on_trackbar)

### Save video
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('thrash/output.avi',fourcc, 20.0, (width,height))

accumulated_time = 0
cont = 0
while True:
    
    cont = cont+1
    # Frame from webcam
   # print("antes de cap read")
    ret, frame = cap.read() 
    if ret:
        
        # Flip frame horizontally
        #print("antes de cap flip")
        frame = cv2.flip(frame, 1)
        
        #%% Gazing
        # Update gazeTracker
        start = time.time()
        try:
            shown_img,_,_ = gazeTracker.Update(frame)
            # out.write(shown_img)
        except Exception as e:
            print("Excepcion al tanto")
            print(str(e))
            break
        accumulated_time = accumulated_time+(time.time()-start)
    
        # Check for calibration pressed, 'c' key
        if (cv2.waitKey(1) & 0xFF == ord('c')):
            gazeTracker.Calibrate()
            
            
        # # If user want's to check blinking! 'b' key
        # if (cv2.waitKey(1) & 0xFF == ord('b')):
        #     gazeTracker.CheckBlinkings()
            
            
        
        #%% Showing
        
        cv2.imshow('vis', shown_img)
        #cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        #cv2.moveWindow(windowName, 0,0)
        
    # To scape press "q" 
    if (cv2.waitKey(1) & 0xFF == ord('q')): 
        print("user exit")
        break


#cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_NORMAL)
cap.release()
# out.release()
gazeTracker.Close()

cv2.destroyAllWindows()
cv2.waitKey(1)


print("AVERAGE Processing time: %0.3f" % (accumulated_time/cont))
print("AVERAGE FPS: %d" % (cont/accumulated_time))
# print("--------------------------------------")
# print("AVERAGE All Detection time: %0.6f s" % (gazeTracker.accum_detection/cont))
# print("        AVERAGE Mediapipe Detection time: %0.6f s" % (gazeTracker.detector.accum_mediapipe/cont))
# print("        AVERAGE Mediapipe Post-Process time: %0.6f s" % (gazeTracker.detector.accum_postmediapipe/cont))
# print("        AVERAGE ONNX Detection time: %0.6f s" % (gazeTracker.detector.accum_onnx/cont))
# print("        AVERAGE ONNX drawing time: %0.6f s" % (gazeTracker.detector.accum_onnxdrawing/cont))
# print("AVERAGE Gaze Vector calculation time: %0.6f s" % (gazeTracker.accum_gazeVector/cont))
# print("AVERAGE Head Pose Estimation time: %0.6f s" % (gazeTracker.accum_headPose/cont))
# print("AVERAGE Final Resizing time: %0.6f s" % (gazeTracker.accum_resize/cont))
