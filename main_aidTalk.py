#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:31:25 2021

@author: josesolla
"""



from PySide2.QtWidgets import QApplication, QDialog, QStackedWidget, QSizePolicy, QDesktopWidget
from PySide2.QtGui import QImage, QPixmap, QCursor
from PySide2.QtCore import QPropertyAnimation, QPoint, Qt, QTimer, QCoreApplication

from PySide2.QtCore import Signal, Slot, QThread
from PySide2.QtTextToSpeech import QTextToSpeech


import sys
import cv2
import numpy as np
import argparse
import time
import telegram_send
import depthai as dai

## UIs
from user_interfaces import * #Ui_frameShowing, Ui_calibrationWindow, Ui_menuWindow, Ui_writingWindow

# Bring Gaze Tracker object!
from gaze_tracker import GazeTracker

# import tkinter as tk

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FOCAL_LENGTH_CM = 0.452
FOCAL_LENGTH_PX = 601.08721
WEBCAM_SOURCE = 0
MONITOR_SOURCE = 0
VOICE = 0


### Create an optional argument parser

parser = argparse.ArgumentParser(description='aidTalk system: from eye to voice')
parser.add_argument("-c", "--camera", nargs='+', type=int, required=False, 
                    help="camera resolution in px: Focal_Width Focal_Height", default=(CAMERA_WIDTH, CAMERA_HEIGHT))
parser.add_argument("-F", "--focal_cm", type=float, required=False, 
                    help="focal length of the webcam in cm", default=FOCAL_LENGTH_CM)
parser.add_argument("-f", "--focal_px", type=float, required=False, 
                    help="focal length of the webcam in px", default=FOCAL_LENGTH_PX)
parser.add_argument("-w", "--webcam", type=int, required=False, 
                    help="webcam source to obtain frames from", default=0)
parser.add_argument("-m", "--monitor", type=int, required=False, 
                    help="monitor source to display program", default=0)
parser.add_argument("-v", "--voice", type=int, required=False, 
                    help="voice to use on TTS: Female is 0, Male is 1", default=0)
values = parser.parse_args()

# Set values to parsed ones
CAMERA_WIDTH = values.camera[0]
CAMERA_HEIGHT = values.camera[1]
FOCAL_LENGTH_CM = values.focal_cm
FOCAL_LENGTH_PX = values.focal_px
WEBCAM_SOURCE = values.webcam
MONITOR_SOURCE = values.monitor
VOICE = values.voice




DOT_LIST = np.array([[1/2, 1/2], ## Initial dot ----- (x,y)
                      [0.05, 0.05], [0.50, 0.95], [0.95, 0.05], # Calibraiton dots
                      [0.50, 0.05], [0.95, 0.95], [0.05, 0.95], 
                      [0.50, 0.50], [0.75, 0.25], [0.25, 0.75], 
                      [0.75, 0.75], [0.25, 0.25]])


WINDOWS = {'init': 0, 'explanation': 1, 'menu': 2, 'calibration': 3, 'writing': 4, 'showing': 5, 'close': 6}


"""
    Thread classes section
"""


class ImageProcessingThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    calibration_finished_signal = Signal()

    def __init__(self, screen):
        super().__init__()
        self._run_flag = True
        self.accumulated_time = 0
        self.accumulated_cont = 0
        
        self.screen = screen
        self.display_width = self.screen.size().width() #width
        self.display_height = self.screen.size().height() #height
        self.gazeTracker = GazeTracker(screen_size=(self.display_width, self.display_height), camera_resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), focal_length_cm=FOCAL_LENGTH_CM, focal_length_px=FOCAL_LENGTH_PX)
        
        self.process_image = True
        self.show_image = False
        
       # self.check_for_calibration = True
        
        self.getMouseControl = False
        self.isGazedControlled = False
        
        #### Control variables and calibration data matrix
        self.new_row = False
        self.append_data = False
        self.calibrate_data = False
        self.calibration_data_matrix_leye = [] # Calibration data por all dots
        self.calibration_data_row_leye = [] # Calibration data for one dot
        self.calibration_data_matrix_reye = [] # Calibration data por all dots
        self.calibration_data_row_reye = [] # Calibration data for one dot

        ### Define a source - color camera
        self.pipeline = dai.Pipeline()
        self.cam_rgb = self.pipeline.createColorCamera()
        self.cam_rgb.setPreviewSize(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        ### Create output
        self.xout_rgb = self.pipeline.createXLinkOut()
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)
        
        # self.window = QWindow()
        
        # self.window.setScreen(self.screen)
        

    def run(self):
        with dai.Device(self.pipeline) as device:
            #device.startPipeline()
            vid = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) 
            
            while self._run_flag:
                # capture from web cam
                cap = vid.get()
                
                if cap is not None:
                    cv_img = cap.getCvFrame()
                    #print(cv_img.shape)
                    start = time.time()
                    #starttt = time.time()
                    cv_img = cv2.flip(cv_img, 1)
                    
                    if self.process_image:
                        
                        #print(QCursor.pos())
                        try:
                            
                            cv_img, mouseX, mouseY = self.gazeTracker.Update(cv_img)
                            #print((mouseX, mouseY))
                            
                            if self.new_row:
                                self.calibration_data_matrix_leye.append(self.calibration_data_row_leye)
                                #print(self.calibration_data_row_leye)
                                self.calibration_data_row_leye = []                            
                                self.calibration_data_matrix_reye.append(self.calibration_data_row_reye)
                                self.calibration_data_row_reye = []
                                #print("Fila nueva")
                                # print("--")
                                # print(self.calibration_data_matrix_leye)
                                #print("------")
                                #print(self.calibration_data_matrix_reye)
                                self.new_row = False
                                self.append_data = True
                            
                            if self.append_data:
                                # print("Nuevos datos a la fila")
                                self.calibration_data_row_leye.append(self.gazeTracker.gaze_vector_leye)
                                self.calibration_data_row_reye.append(self.gazeTracker.gaze_vector_reye)
                            
                            
                            if self.calibrate_data:
                                
                                ## Eliminate empty lists (at the beggining there's always one)
                                self.calibration_data_matrix_leye = [x for x in self.calibration_data_matrix_leye if x]
                                self.calibration_data_matrix_reye = [x for x in self.calibration_data_matrix_reye if x]
                                
                                leye_calibration_data = filterData(self.calibration_data_matrix_leye)
                                reye_calibration_data = filterData(self.calibration_data_matrix_reye)
                                
                                # print(leye_calibration_data)
                                # print(reye_calibration_data)
                                
                                self.gazeTracker.calib_coeffs_leye = self.gazeTracker.calibrator.mapGaze(leye_calibration_data)
                                self.gazeTracker.calib_coeffs_reye = self.gazeTracker.calibrator.mapGaze(reye_calibration_data)
                                self.gazeTracker.calibrator.calibrated=True
                                self.calibrate_data = False
                                
                                # Re-start variables
                                self.calibration_data_matrix_leye = [] # Calibration data por all dots
                                self.calibration_data_row_leye = [] # Calibration data for one dot
                                self.calibration_data_matrix_reye = [] # Calibration data por all dots
                                self.calibration_data_row_reye = [] 
                                self.new_row = False
                                self.append_data = False
                                self.calibrate_data = False
                                #print("wiiiiiiiiiii")
    
                            if self.getMouseControl:
                                if mouseX is not None and mouseY is not None:
                                    #moove mouse cursor
                                    #QCursor.setPos(int(mouseX), int(mouseY))
                                    if MONITOR_SOURCE != 0:
                                        QCursor.setPos(int(mouseX-self.display_width), int(mouseY))
                                    else:
                                        QCursor.setPos(int(mouseX), int(mouseY))
                                    #self.moveCursor(int(mouseX), int(mouseY))
                                    #QCursor.setPos(self.screen, QPoint(int(mouseX), int(mouseY)))
                            #print(QCursor.pos(self.screen))
                        except Exception as err:
                            print(str(err))
                            # print("hubo excepcion")
                            pass
                    
                    
                    self.accumulated_time =  self.accumulated_time + (time.time()-start)
                    self.accumulated_cont = self.accumulated_cont+1
                    if self.show_image: self.change_pixmap_signal.emit(cv_img)
            # shut down capture system
            
            print("AVERAGE Processing time: %0.3f" % (self.accumulated_time/self.accumulated_cont))
            print("AVERAGE FPS: %d" % (self.accumulated_cont/self.accumulated_time))
    
    
    def moveCursor(self, x, y):
        
        #sscreen = QGuiApplication.primaryScreen()
        #print(QGuiApplication.screens())
        cursor = QCursor()
        cursor.setPos(x, y)
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


def filterData(data, n_std_times=1):
    
    final_data = []
    for dot_data in data:
        dot_data = np.array(dot_data)
        mean = np.mean(dot_data, axis=0)
        std  = np.std(dot_data, axis=0)
        sup_lim = mean+std*n_std_times
        inf_lim = mean-std*n_std_times
        #outliers = np.where(np.logical_and(np.logical_or(brbr[:,0]<inf_lim[0], brbr[:,0]>sup_lim[0]), np.logical_or(brbr[:,1]<inf_lim[1], brbr[:,1]>sup_lim[1])))
        inliers = np.where(np.logical_or(np.logical_and(dot_data[:,0]>inf_lim[0], dot_data[:,0]<sup_lim[0]), np.logical_and(dot_data[:,1]>inf_lim[1], dot_data[:,1]<sup_lim[1])))[0]
        
        final_data.append(np.mean(dot_data[inliers, :], axis=0))
        
    return np.array(final_data)

"""
    GUI classes section. 
    
    0 is First Window (Menu Now)
    1 is Gaze Tracking (CalibrationWindow)
    2 is KeyBoard (KeyboardGUI)
    3 is frame processing results showing (ShowingWindow)
    
    
"""

## Esto va a ser comodo para cuando tengamos todas en orden
guis_dict = {'initial': 0, 'menu': 1, 'calibration': 2, 'keyboard': 3, 'frameShowing': 4}


class MenuWindow(QDialog):
    
    def __init__(self, mainClass):
        super(MenuWindow, self).__init__()
        self.ui = Ui_menuWindow()
        self.ui.setupUi(self)

        # To be able to acces aidTalkApp attributes and methods
        self.parent = mainClass
        
        self.ui.PB_Gaze.clicked.connect(self.goToGaze)
        self.ui.PB_Head.clicked.connect(self.goToHead)


        
    
    @Slot()
    def goToGaze(self):
        #self.parent.setCurrentIndex(1)
        self.parent.thread.gazeTracker.GazeControlled()
        self.parent.thread.isGazedControlled = True
        self.parent.setCurrentIndex(WINDOWS['calibration'])
        time = self.parent.calibrationWindow.initial_waiting_time
        self.parent.calibrationWindow.initial_timer.start(time)
        
    @Slot()
    def goToHead(self):
        self.parent.thread.getMouseControl = True
        self.parent.thread.gazeTracker.HeadControlled()
        self.parent.thread.isGazedControlled = False
        self.parent.setCurrentIndex(WINDOWS['writing'])
        #self.parent.thread.show_image = True


    
    


### Calibration GUI loading
class CalibrationWindow(QDialog):
    
    def __init__(self, mainClass):
        super(CalibrationWindow, self).__init__()
        self.ui = Ui_calibrationWindow()
        self.ui.setupUi(self)
        # To be able to acces aidTalkApp attributes and methods
        self.parent = mainClass
        
        # Positions for dots during calibration
        self.positionsList = ((DOT_LIST[1:,:]*[self.parent.screen.size().width(), 
                                               self.parent.screen.size().height()]) - 30).astype(int).tolist()
        
        ## Move central opening widget to center
        self.ui.calibrationDotBase.move(self.parent.middle[0], self.parent.middle[1])
        self.ui.label.setText("Stare firmly at the center of the dots")
        x = self.parent.middle[0]+30-int(self.ui.label.size().width()/2)
        y = self.parent.middle[1]-30-int(self.ui.label.size().height()/2)
        self.ui.label.move(x, y)
        
        # Animation
        self.animation = QPropertyAnimation(self.ui.calibrationDotBase, b"pos")
        self.animation.finished.connect(self.newDotCapture)

        ##### Timers
        self.initial_timer = QTimer()
        self.initial_timer.timeout.connect(self.startCalibration)
        
        self.sequential_timer = QTimer()
        self.sequential_timer.timeout.connect(self.moveDot)
        
        self.sequential_pretimer = QTimer()
        self.sequential_pretimer.timeout.connect(self.startCapture)

        self.initial_waiting_time = 5000
        self.animation_duration = 300
        self.pretimer_time = 500
        self.capturing_time = 2500
        
        # Number of dots control variable
        self.dot_position_index = -1
        
    
    @Slot()
    def moveDot(self):
        self.dot_position_index += 1
        
        self.sequential_timer.stop()
        self.parent.thread.append_data = False
        if self.dot_position_index >= len(self.positionsList):
            
            self.dot_position_index = 0
            self.changeWindow()
        
        else:
            
            i = self.dot_position_index
            self.animation.setEndValue(QPoint(self.positionsList[i][0],self.positionsList[i][1]))
            self.animation.setDuration(300)
            self.animation.start()
            
        
    
    ## Wait 0.5 seconds before capturing to avoid micro-saccades outliers
    @Slot()
    def newDotCapture(self):
        
        self.sequential_pretimer.start(500)
        
    
    @Slot()
    def startCapture(self):
        
        #print(self.dot_position_index)
        self.sequential_pretimer.stop()
        # Set to true capture of gaze Vectors
        self.parent.thread.new_row = True
        #self.parent.thread.append_data = True
        self.sequential_timer.start(2500)
    
    ######## Related to animations
    
    ## Stop initial timer and start calibration timers
    @Slot()
    def startCalibration(self):
        self.initial_timer.stop()
        #self.parent.setCurrentIndex(3) ## Comentar esto y luego descomentar lo de abaixo
        self.ui.label.setText(" ") # Delete label text
        
        # First animation
        self.moveDot()

    @Slot()
    def changeWindow(self):
        # Stop capturing gaze vectors and compute mapping function
        self.parent.thread.append_data = False
        self.parent.thread.new_row = True
        self.parent.thread.calibrate_data = True
        self.parent.setCurrentIndex(WINDOWS['writing']) # De momento que mande a la visualizacion, pero al 2 esta bien por el keyboard
        self.parent.thread.getMouseControl = True
        self.dot_position_index = -1
    

        

class ShowingWindow(QDialog):
    
    def __init__(self, mainClass):
        super(ShowingWindow, self).__init__()
        self.ui = Ui_frameShowing()
        self.ui.setupUi(self)
        
        # To be able to acces aidTalkApp attributes and methods
        self.parent = mainClass
        
        self.display_width = self.parent.screen.size().width() #width
        self.display_height = self.parent.screen.size().height() #height
        
        self.ui.image_label.resize(self.display_width, self.display_height)
        self.ui.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.image_label.setAlignment(Qt.AlignCenter)
        
 
    
    @Slot()
    def calibrationDone(self):
        self.parent.setCurrentIndex(WINDOWS['writing'])
    
    
    @Slot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        
        qt_img = self.convert_cv_qt(cv_img)
        self.ui.image_label.setPixmap(qt_img)
        #pass
    
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    
        

class WritingWindow(QDialog):
    def __init__(self, mainClass):
        super(WritingWindow, self).__init__()
        self.ui = Ui_writingWindow()
        self.ui.setupUi(self)
        
        self.qtts = QTextToSpeech()
        self.qtts.setVoice(self.qtts.availableVoices()[VOICE])
        #print(self.qtts.availableLocales()) # setLocale para decir el idioma, por defecto coge el del sistema
        
        self.parent = mainClass
        
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.write)
        self.timer_time = 500 #ms
   
        self.robin_index = -1
        
        self.key_index = None
        self.last_key_index = None
        self.text2speech = ''
        self.dic = [['a','b','c'],
                    ['d','e','f'],
                    ['g','h','i'],
                    ['j','k','l'],
                    ['m','n','o','ñ'],
                    ['p','q','r','s'],
                    ['t','u','v'],
                    ['w','x','y','z'],
                    [' ','?','!', ',']]
        
        self.ui.pushButton.clicked.connect(self.b_clicked)
        self.ui.pushButton1.clicked.connect(self.b1_clicked)
        self.ui.pushButton2.clicked.connect(self.b2_clicked)
        self.ui.pushButton3.clicked.connect(self.b3_clicked)
        self.ui.pushButton4.clicked.connect(self.b4_clicked)
        self.ui.pushButton5.clicked.connect(self.b5_clicked)
        self.ui.pushButton6.clicked.connect(self.b6_clicked)
        self.ui.pushButton7.clicked.connect(self.b7_clicked)
        self.ui.pushButton8.clicked.connect(self.b8_clicked)
        self.ui.pushButton9.clicked.connect(self.b9_clicked)
        self.ui.pushButton10.clicked.connect(self.b10_clicked)
        self.ui.pushButton11.clicked.connect(self.b11_clicked)
        self.ui.pushButton12.clicked.connect(self.b12_clicked)
        self.ui.pushButton13.clicked.connect(self.b13_clicked)
    	

    #     self.ui.pushButton1.clicked.connect(self.entra)
    #     self.ui.pushButton10.clicked.connect(self.sal)
    
    # @Slot()    
    # def entra(self):

    # 	self.timerrr = time.time()

    # @Slot()    
    # def sal(self):

    # 	print(time.time()-self.timerrr)

    
    @Slot()    
    def write(self):
        
        ###writing intreface control  
        #print(self.timer.remainingTime())
        if self.key_index is not None:
            if self.key_index == 5 or self.key_index == 7 or self.key_index == 8 or self.key_index == 4:
                self.robin_index%=4
            
            else:
                self.robin_index%=3
                    
            self.text2speech = self.text2speech + self.dic[self.key_index][self.robin_index]
            #print(self.text2speech)
            
            self.ui.textEdit.setText(self.text2speech)
            self.robin_index = -1    
            self.timer.stop()
    
    ## Telegram button
    @Slot()
    def b_clicked(self):
        telegram_send.send(messages=["Please come to my room"])
        #self.text2speech = ''
        #self.ui.textEdit.setText(self.text2speech)

    ## Writing buttons
    @Slot()
    def b1_clicked(self):
       
        # # Check if timer is still active if last pressed key was the same (write previous letter without waiting for timer)
        if self.timer.remainingTime()>0 and self.key_index!=0:
            self.write()
  
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 0
       
        
    @Slot()   
    def b2_clicked(self):
               
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=1:
            self.write()
            
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 1
        
    @Slot()
    def b3_clicked(self):
        
        # # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=2:
            self.write()
        
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 2
        
    @Slot()    
    def b4_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=3:
            self.write()
       
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 3
        
    @Slot()    
    def b5_clicked(self):
        
        # # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=4:
            self.write()
  
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 4
        
    @Slot()    
    def b6_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=5:
            self.write()
        
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 5
        
    @Slot()    
    def b7_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=6:
            self.write()
        
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 6
        
    @Slot()    
    def b8_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=7:
            self.write()
      
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 7
        
    @Slot()    
    def b9_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=8:
            self.write()
      
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 8
    
    ## tts button
    @Slot()    
    def b10_clicked(self):
 
        self.qtts.say(self.text2speech)
        self.text2speech = ''
        self.ui.textEdit.setText(self.text2speech)
        
    ## delete button
    @Slot()
    def b11_clicked(self):
        
        self.text2speech = self.text2speech[:-1]
        self.ui.textEdit.setText(self.text2speech)
        #print(self.text2speech)
    
    ## close button
    @Slot()
    def b12_clicked(self):
        self.parent.setCurrentIndex(WINDOWS['close'])
        self.parent.closeWindow.close()
        
    
    ## go to menu button
    @Slot()
    def b13_clicked(self):
        self.parent.thread.getMouseControl = False
        self.parent.setCurrentIndex(WINDOWS['menu'])
    


class InitWindow(QDialog):

    def __init__(self, mainClass):
        super(InitWindow, self).__init__()
        self.ui = Ui_inicio()
        self.ui.setupUi(self)
        
        self.parent = mainClass
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.changeWindow)
        self.timer.start(6000) # 6 seconds
        
    @Slot()
    def changeWindow(self):
        self.timer.stop()
        self.parent.setCurrentIndex(WINDOWS['explanation'])
        


class ExplanationWindows(QDialog):
    def __init__(self, mainClass):
        super(ExplanationWindows, self).__init__()
        self.ui = Ui_explicacion1()
        self.ui.setupUi(self)
        
        self.parent = mainClass
        
        self.translate = QCoreApplication.translate
        
        self.expl1_text = "<html><head/><body><p align=\"justify\"><span style=\" font-size:18pt; font-weight:600;\">aidTalk</span><span style=\" font-size:18pt;\"> Is a gaze and head controlled text-2-speech system cenceived to facilitate large dependents to communicate in a fluid way.</span></p><p align=\"justify\"><br/></p><p align=\"justify\"><span style=\" font-size:18pt; font-weight:600;\">aidTalk </span><span style=\" font-size:18pt;\">allows cursor control over an interface designed to facilitate key seleccion. This keyboard control can be done in two different ways:</span></p><p align=\"justify\"><br/></p><p align=\"center\"><span style=\" font-size:24pt; font-weight:600;\">Gaze Control</span></p><p align=\"center\"><br/></p><p align=\"center\"><span style=\" font-size:24pt; font-weight:600;\">Head control</span></p><span style=\" font-size:16pt; font-weight:300;\">Press: C to continue, M to Menu, Space to exit</span></p></body></html>"
        self.expl2_text = "<html><head/><body><p align=\"center\"><span style=\" font-size:24pt; font-weight:600;\">Gaze control</span></p><p align=\"center\"><br/></p><p align=\"justify\"><span style=\" font-size:18pt;\">A calibration screen will be shown to match the user´s gaze vectors to the monitor dimensions. During this process is completely necessary to stare at the center of the dots shown in the screen. For optimal results head must remain still during the calibration process and system use.</span></p><p align=\"justify\"><br/></p><p align=\"justify\"><span style=\" font-size:18pt;\">When the system is calibrated, it is ready to communicate. Remember, any head movemente will cause and error in the gaze controll and calibration will be lost.</span></p></body></html>"
        self.expl3_text = "<html><head/><body><p align=\"center\"><span style=\" font-size:24pt; font-weight:600;\">Head control</span></p><p align=\"center\"><br/></p><p align=\"justify\"><span style=\" font-size:18pt;\">Head tracking will allow the gaze control system work with free head movement in future versions. In the meantime,it offers the possibility to control the cursor pointing with the head towards the desired point in the screen.</span></p><p align=\"justify\"><br/></p><p align=\"justify\"><span style=\" font-size:18pt;\">With this control it is not necessary to calibrate the system, so, it is ready to use by clicking.</span></p></body></html>"
        self.expl4_text = "<html><head/><body><p align=\"center\"><span style=\" font-size:24pt; font-weight:600;\">Write confortably!</span></p><p align=\"center\"><br/></p><p align=\"justify\"><span style=\" font-size:18pt;\">The key point of te system is to be inclusive and accessible for every person who could need it. This is why this project is equipped with multiple sensors, each of them covering the needs of some especific user. All this sensors are designed using and Arduino Leonardo, so, HID libraries can be used to generate the clicks the system need to work. Whit this method, is easier to adapt the sensor to the user´s especific needs.</span></p><p align=\"justify\"><span style=\" font-size:18pt;\">Since it is difficult to provide this demo with sensors, use your mouse or trackpad to generate the clicks needed.</span></p></body></html>"
        self.expl5_text = "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt; font-weight:600;\">Enjoy AidTalk!</span></p><p align=\"center\"><span style=\" font-size:36pt;\"><br/></span></p><p align=\"center\"><span style=\" font-size:36pt;\"><br/></span></p></body></html>"
        
        
        self.text_list = [self.expl1_text, self.expl2_text, self.expl3_text, 
                          self.expl4_text, self.expl5_text]
        self.cont = 0
    
        
        
class CloseWindow(QDialog):

    def __init__(self, mainClass):
        super(CloseWindow, self).__init__()
        self.ui = Ui_cierre()
        self.ui.setupUi(self)
        
        self.parent = mainClass
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.exitProgram)
       
    
    def close(self):
        self.parent.thread.getMouseControl = False
        self.timer.start(2000) # 4 seconds
    
    @Slot()
    def exitProgram(self):
        self.timer.stop()
        self.parent.closeApp()

class aidTalkApp(QStackedWidget):   
    
    def __init__(self, selectedScreenObject):
        super(aidTalkApp, self).__init__()
        
        # Screen options
        
        self.selected_screen = MONITOR_SOURCE
        #print(selectedScreenObject.size().width(), selectedScreenObject.size().height())
        self.screen = selectedScreenObject
        #self.setScreen(self.screen)
        #print(self.screen)
        # Middel psotion of the screen widget
        self.middle = ((DOT_LIST[0,:]*[self.screen.size().width(), 
                                       self.screen.size().height()]) - 30).astype(int).tolist()
        

        ##### Stackear Ventanucas aqui
        self.initWindow = InitWindow(self)
        self.addWidget(self.initWindow)
        
        self.explanationWindows = ExplanationWindows(self)
        self.addWidget(self.explanationWindows)
        
        self.menuWindow = MenuWindow(self)
        self.addWidget(self.menuWindow)
        
        self.calibrationWindow = CalibrationWindow(self)
        self.addWidget(self.calibrationWindow)
        
        self.writingWindow = WritingWindow(self)
        self.addWidget(self.writingWindow)
        
        self.showingWindow = ShowingWindow(self)
        self.addWidget(self.showingWindow)
        
        self.closeWindow = CloseWindow(self)
        self.addWidget(self.closeWindow)

        ##### Showing options
        monitor = QDesktopWidget().screenGeometry(self.selected_screen)
        self.move(monitor.left(), monitor.top()) 
        self.showFullScreen()
        #self.showMaximized()
        


        ##### Image processing thread
        self.thread = ImageProcessingThread(self.screen)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.showingWindow.update_image)
        self.thread.calibration_finished_signal.connect(self.showingWindow.calibrationDone)
        # start the thread
        self.thread.start()
        
        
    
    ### Por comodité. Key events
    def keyPressEvent(self, event):
        
        if event.key() == Qt.Key_Space:
            self.setCurrentIndex(WINDOWS['close'])
            self.closeWindow.close()
            
        elif event.key() == Qt.Key_A:
            self.thread.getMouseControl = False
            
            
        elif event.key() == Qt.Key_Up: # Re-calibrate
        
            if self.thread.isGazedControlled:
                self.setCurrentIndex(WINDOWS['calibration'])
                self.thread.getMouseControl = False
                self.calibrationWindow.ui.calibrationDotBase.move(self.middle[0], self.middle[1])
                self.calibrationWindow.ui.label.setText("Stare firmly at the center of the dots")
                time = self.calibrationWindow.initial_waiting_time
                self.calibrationWindow.initial_timer.start(time)
            
        elif event.key() == Qt.Key_Down:
            self.setCurrentIndex(WINDOWS['close'])
            self.closeWindow.close()
           
        # For changing to visualization mode
        elif event.key() == Qt.Key_V:
            self.setCurrentIndex(WINDOWS['showing'])
            self.thread.show_image = True
            self.thread.getMouseControl = False
            
        elif event.key() == Qt.Key_M:
            self.setCurrentIndex(WINDOWS['menu'])
            self.thread.getMouseControl = False
            
        elif event.key() == Qt.Key_C:
            #print("pulsa c")
            self.explanationWindows.cont += 1
            if self.explanationWindows.cont >= 5:
                self.explanationWindows.cont = -1
                # Cambiar de ventanuca
                self.setCurrentIndex(WINDOWS['menu'])
                
            else: 
                self.explanationWindows.ui.label.setText(self.explanationWindows.translate("Dialog", self.explanationWindows.text_list[self.explanationWindows.cont]))

           
    @Slot()
    def closeApp(self):
        self.thread.stop()
        QApplication.instance().quit()
        
    # For thread management
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    

if __name__=="__main__":
    
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    a = aidTalkApp(app.screens()[MONITOR_SOURCE])
    a.show()
    sys.exit(app.exec_())


