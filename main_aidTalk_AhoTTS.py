#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:31:25 2021

@author: josesolla
"""
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QDialog, QMainWindow, QStackedWidget, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QFile, QPropertyAnimation, QPoint, QVariantAnimation, QAbstractAnimation, QSequentialAnimationGroup, Qt, QTimer
from PyQt5.QtGui import QColor, QCursor
from PyQt5.Qt import Qt as qtKeys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread

import sys
import cv2
import numpy as np

#import pyautogui as pg
# Bring Gaze Tracker object!
from gaze_tracker import GazeTracker
import tts
import tkinter as tk


import time

# ## Cross-platform way to obtain screen size. Cuidadín, a mi me rompe el programa y reinicia el kernel
# root = tk.Tk()
# width = root.winfo_screenwidth()
# height = root.winfo_screenheight()
# root.destroy()
# del root

DISPLAY_WIDTH = 1280#1280
DISPLAY_HEIGHT = 800#800
CAMERA_WIDTH = 1280#1280
CAMERA_HEIGHT = 720#800
FOCAL_LENGTH_CM = 0.3*2.1
FOCAL_LENGTH_PX = 1085.54

# positionsList = [[15, 30], [615, 30], [1280-75, 30], 
#                   [15, 345], [615, 345], [1280-75, 345], 
#                   [15, 800-75], [615, 800-75], [1280-75, 800-75]]

DOT_LIST = np.array([[1/2, 1/2], ## Initial dot ----- (x,y)
                      [0.05, 0.05], [0.50, 0.95], [0.95, 0.05], # Calibraiton dots
                      [0.50, 0.05], [0.95, 0.95], [0.05, 0.95], 
                      [0.50, 0.50], [0.75, 0.25], [0.25, 0.75], 
                      [0.75, 0.75], [0.25, 0.25]])

positionsList = ((DOT_LIST[1:,:]*[DISPLAY_WIDTH, DISPLAY_HEIGHT]) - 30).astype(int).tolist()

# positionsList = [[34, 10],
#  [610, 730],
#  [1186, 10],
#  [610, 10],
#  [1186, 730],
#  [34, 730],
#  [610, 370],
#  [930, 170],
#  [290, 570],
#  [930, 570],
#  [290, 170]]


"""
    Thread classes section
"""


class ImageProcessingThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    calibration_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.accumulated_time = 0
        self.accumulated_cont = 0
        self.display_width = DISPLAY_WIDTH #width
        self.display_height = DISPLAY_HEIGHT #height
        self.gazeTracker = GazeTracker(screen_size=(self.display_width, self.display_height), camera_resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), focal_length_cm=FOCAL_LENGTH_CM, focal_length_px=FOCAL_LENGTH_PX)
        
        self.process_image = True
        
       # self.check_for_calibration = True
        
        self.getMouseControl = False
        
        #### Control variables and calibration data matrix
        self.new_row = False
        self.append_data = False
        self.calibrate_data = False
        self.calibration_data_matrix_leye = [] # Calibration data por all dots
        self.calibration_data_row_leye = [] # Calibration data for one dot
        self.calibration_data_matrix_reye = [] # Calibration data por all dots
        self.calibration_data_row_reye = [] # Calibration data for one dot
        
        
        #self.cursor = QCursor()
        

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                #print(cv_img.shape)
                start = time.time()
                #starttt = time.time()
                cv_img = cv2.flip(cv_img, 1)
                
                if self.process_image:
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
                        
                        # if self.check_for_calibration:
                        #     if self.gazeTracker.isCalibrationFinished():
                        #         self.check_for_calibration = False
                        #         self.calibration_finished_signal.emit()
                            
                        if self.getMouseControl:
                            if mouseX is not None and mouseY is not None:
                                #moove mouse cursor
                                QCursor.setPos(int(mouseX), int(mouseY))
                                #pg.moveTo(int(mouseX), int(mouseY))
                                pass
                    except Exception as err:
                        print(str(err))
                        # print("hubo excepcion")
                        pass
                
                
                self.accumulated_time =  self.accumulated_time + (time.time()-start)
                self.accumulated_cont = self.accumulated_cont+1
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()
        print("AVERAGE Processing time: %0.3f" % (self.accumulated_time/self.accumulated_cont))
        print("AVERAGE FPS: %d" % (self.accumulated_cont/self.accumulated_time))

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
        super().__init__()
        uic.loadUi("layouts/menuWindow.ui", self) 
        
        # To be able to acces aidTalkApp attributes and methods
        self.parent = mainClass
        
        self.PB_Gaze.clicked.connect(self.goToGaze)
        self.PB_Head.clicked.connect(self.goToHead)

    @pyqtSlot()
    def goToGaze(self):
        #self.parent.setCurrentIndex(1)
        self.parent.showingFrame = True 
        self.parent.thread.gazeTracker.GazeControlled()
        #self.parent.thread.gazeTracker.Calibrate()
        self.parent.setCurrentIndex(1)
        time = self.parent.calibrationWindow.initial_waiting_time
        self.parent.calibrationWindow.initial_timer.start(time)
        #self.parent.calibrationWindow.startAnimationThread()
        
    @pyqtSlot()
    def goToHead(self):
        self.parent.showingFrame = True 
        self.parent.thread.getMouseControl = True
        self.parent.thread.gazeTracker.HeadControlled()
        self.parent.setCurrentIndex(2)
    
    


### Calibration GUI loading
class CalibrationWindow(QDialog):
    
    def __init__(self, mainClass):
        super().__init__()
        uic.loadUi("layouts/calibrationGUI.ui", self) #Cargar form.ui
        
        # To be able to acces aidTalkApp attributes and methods
        self.parent = mainClass
        
        # # Dot animation sequence
        # self.anim_group = QSequentialAnimationGroup()
        # self.anim_group.currentAnimationChanged.connect(self.animationChange)
        # self.anim_group.finished.connect(self.changeWindow)
        
        
        self.animation = QPropertyAnimation(self.calibrationDotBase, b"pos")
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
        
        #self.current_animation = -1 #1 for moving dot, 2 for moving center
        
    
    ######## Related to animation changes and capturing gaze vectors
    
    @pyqtSlot()
    def moveDot(self):
        self.dot_position_index += 1
        
        self.sequential_timer.stop()
        self.parent.thread.append_data = False
        if self.dot_position_index >= len(positionsList):
            
            self.dot_position_index = 0
            self.changeWindow()
        
        else:
            
            i = self.dot_position_index
            self.animation.setEndValue(QPoint(positionsList[i][0],positionsList[i][1]))
            self.animation.setDuration(300)
            self.animation.start()
            
        
    
    ## Wait 0.5 seconds before capturing to avoid micro-saccades outliers
    @pyqtSlot()
    def newDotCapture(self):
        
        self.sequential_pretimer.start(500)
        
    
    @pyqtSlot()
    def startCapture(self):
        
        print(self.dot_position_index)
        self.sequential_pretimer.stop()
        # Set to true capture of gaze Vectors
        self.parent.thread.new_row = True
        #self.parent.thread.append_data = True
        self.sequential_timer.start(2500)
    
    ######## Related to animations
    
    ## Stop initial timer and start calibration timers
    @pyqtSlot()
    def startCalibration(self):
        self.initial_timer.stop()
        #self.parent.setCurrentIndex(3) ## Comentar esto y luego descomentar lo de abaixo
        self.label.setText(" ") # Delete label text
        
        # First animation
        self.moveDot()

    @pyqtSlot()
    def changeWindow(self):
        # Stop capturing gaze vectors and compute mapping function
        self.parent.thread.append_data = False
        self.parent.thread.new_row = True
        self.parent.thread.calibrate_data = True
        self.parent.showingFrame = True 
        self.parent.setCurrentIndex(2) # De momento que mande a la visualizacion, pero al 2 esta bien por el keyboard
        self.parent.thread.getMouseControl = True
    

        

class ShowingWindow(QDialog):
    
    def __init__(self, mainClass):
        super().__init__()
        uic.loadUi("layouts/frameShowing.ui", self)
        
        # To be able to acces aidTalkApp attributes and methods
        self.parent = mainClass
        
        self.display_width = DISPLAY_WIDTH #width
        self.display_height = DISPLAY_HEIGHT #height
        
        self.image_label.resize(self.display_width, self.display_height)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        
    
    @pyqtSlot()
    def calibrationDone(self):
        self.parent.setCurrentIndex(2)
    
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        
        if self.parent.showingFrame:
            qt_img = self.convert_cv_qt(cv_img)
            self.image_label.setPixmap(qt_img)
        #pass
    
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    
        

class WritingWindow(QDialog):
    def __init__(self, mainClass):
        super().__init__()
        uic.loadUi("layouts/writing.ui", self)
        
        self.tts = tts.AhoTTS2()
        
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
                    ['m','n','o'],
                    ['p','q','r','s'],
                    ['t','u','v'],
                    ['w','x','y','z'],
                    [' ','?','!']]
        
        self.pushButton1.clicked.connect(self.b1_clicked)
        self.pushButton2.clicked.connect(self.b2_clicked)
        self.pushButton3.clicked.connect(self.b3_clicked)
        self.pushButton4.clicked.connect(self.b4_clicked)
        self.pushButton5.clicked.connect(self.b5_clicked)
        self.pushButton6.clicked.connect(self.b6_clicked)
        self.pushButton7.clicked.connect(self.b7_clicked)
        self.pushButton8.clicked.connect(self.b8_clicked)
        self.pushButton9.clicked.connect(self.b9_clicked)
        self.pushButton10.clicked.connect(self.b10_clicked)
        self.pushButton11.clicked.connect(self.b11_clicked)
        self.pushButton12.clicked.connect(QApplication.instance().quit)
        self.pushButton13.clicked.connect(self.b13_clicked)
        
    @pyqtSlot()    
    def write(self):
        
        ###writing intreface control  
        #print(self.timer.remainingTime())
        if self.key_index is not None:
            if self.key_index == 5 or self.key_index == 7:
                self.robin_index%=4
            
            else:
                self.robin_index%=3
                    
            self.text2speech = self.text2speech + self.dic[self.key_index][self.robin_index]
            #print(self.text2speech)
            
            self.textEdit.setText(self.text2speech)
            self.robin_index = -1    
            self.timer.stop()
        
    @pyqtSlot()
    def b1_clicked(self):
       
        # # Check if timer is still active if last pressed key was the same (write previous letter without waiting for timer)
        if self.timer.remainingTime()>0 and self.key_index!=0:
            self.write()
  
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 0
       
        
    @pyqtSlot()   
    def b2_clicked(self):
               
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=1:
            self.write()
            
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 1
        
    @pyqtSlot()
    def b3_clicked(self):
        
        # # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=2:
            self.write()
        
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 2
        
    @pyqtSlot()    
    def b4_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=3:
            self.write()
       
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 3
        
    @pyqtSlot()    
    def b5_clicked(self):
        
        # # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=4:
            self.write()
  
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 4
        
    @pyqtSlot()    
    def b6_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=5:
            self.write()
        
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 5
        
    @pyqtSlot()    
    def b7_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0and self.key_index!=6:
            self.write()
        
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 6
        
    @pyqtSlot()    
    def b8_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=7:
            self.write()
      
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 7
        
    @pyqtSlot()    
    def b9_clicked(self):
        
        # Check if timer is still active if last pressed key was the same
        if self.timer.remainingTime()>0 and self.key_index!=8:
            self.write()
      
        self.timer.start(self.timer_time)
        #self.thread.set_robin_time = True
        #self.thread.start()
        self.robin_index += 1
        self.key_index = 8
        
    @pyqtSlot()    
    def b10_clicked(self):
        """
        Under construction, to send self.thread.text2speech to the tts

        """
        self.tts.tts(self.text2speech)
        self.text2speech = ''
        pass
    
    @pyqtSlot()
    def b11_clicked(self):
        
        self.text2speech = self.text2speech[:-1]
        self.textEdit.setText(self.text2speech)
        #print(self.text2speech)
       
    # @pyqtSlot()
    # def b12_clicked(self):
    #     pass
    
    @pyqtSlot()
    def b13_clicked(self):
        self.parent.setCurrentIndex(0)
    
    



class aidTalkApp(QStackedWidget):   
    
    def __init__(self):
        super().__init__()
        
        self.showingFrame = False
        
        ##### Stackear Ventanucas aqui
        self.menuWindow = MenuWindow(self)
        self.addWidget(self.menuWindow)
        self.calibrationWindow = CalibrationWindow(self)
        self.addWidget(self.calibrationWindow)
        self.writingWindow = WritingWindow(self)
        self.addWidget(self.writingWindow)
        self.showingWindow = ShowingWindow(self)
        self.addWidget(self.showingWindow)

        ##### Showing options
        #self.move(0, 0)
        self.showFullScreen()
        #self.showMaximized()
        


        ##### Image processing thread
        self.thread = ImageProcessingThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.showingWindow.update_image)
        self.thread.calibration_finished_signal.connect(self.showingWindow.calibrationDone)
        # start the thread
        self.thread.start()
        
    
    
    ### Por comodité
    def keyPressEvent(self, event):
        if event.key() == qtKeys.Key_Space:
            QApplication.instance().quit()
            
        elif event.key() == qtKeys.Key_A:
            self.thread.getMouseControl = False
            
        elif event.key() == qtKeys.Key_Up: # Re-calibrate
            
            #if self.thread.
            self.setCurrentIndex(1)
            time = self.parent.calibrationWindow.initial_waiting_time
            self.calibrationWindow.initial_timer.start(time)
            
        elif event.key() == qtKeys.Key_Down:
           QApplication.instance().quit()
            
    
#     def keyPressEvent(self, event):
#         event.key() == Qt.Key_Space:
# 			QApplication.instance().quit()
        
    
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    
        


if __name__=="__main__":
    app = QApplication(sys.argv)
    a = aidTalkApp()
    a.show()
    sys.exit(app.exec_())