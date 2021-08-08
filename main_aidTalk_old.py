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

import pyautogui as pg
# Bring Gaze Tracker object!
from gaze_tracker import GazeTracker
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

# positionsList = [[15, 30], [615, 30], [1280-75, 30], 
#                   [15, 345], [615, 345], [1280-75, 345], 
#                   [15, 800-75], [615, 800-75], [1280-75, 800-75]]
positionsList = [[34, 10],
 [610, 730],
 [1186, 10],
 [610, 10],
 [1186, 730],
 [34, 730],
 [610, 370],
 [930, 170],
 [290, 570],
 [930, 570],
 [290, 170]]


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
        self.gazeTracker = GazeTracker(screen_size=(self.display_width, self.display_height), camera_resolution=(CAMERA_WIDTH, CAMERA_HEIGHT))
        
        self.process_image = True
        
        self.check_for_calibration = True
        
        self.getMouseControl = False
        

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
                        
                        if self.check_for_calibration:
                            if self.gazeTracker.isCalibrationFinished():
                                self.check_for_calibration = False
                                self.calibration_finished_signal.emit()
                            
                        if self.getMouseControl:
                            if mouseX is not None and mouseY is not None:
                                #moove mouse cursor
                                pg.moveTo(int(mouseX), int(mouseY))
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
        self.parent.thread.getMouseControl = True
        self.parent.thread.gazeTracker.GazeControlled()
        self.parent.thread.gazeTracker.Calibrate()
        self.parent.setCurrentIndex(3)
        #self.parent.calibrationWindow.initial_timer.start(5000)
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
        
        # Dot animation sequence
        self.anim_group = QSequentialAnimationGroup()
        self.anim_group.currentAnimationChanged.connect(self.animationChange)
        self.anim_group.finished.connect(self.changeWindow)


        ##### Timers
        self.initial_timer = QTimer()
        self.initial_timer.timeout.connect(self.startCalibration)
        
        
        # Number of dots control variable
        self.dot_position_index = 0
        
        self.current_animation = -1 #1 for moving dot, 2 for moving center
        
    
    ######## Related to animation changes and capturing gaze vectors
    
    ##### Hay que decirle al thread que:
        # Si 1, capture gazevectors en fila
        # Si 2, cree nueva fila
    
    @pyqtSlot()
    def animationChange(self):
        
        if self.current_animation == 1:
            self.dot_position_index +=1
            self.current_animation = 2
            self.parent.thread.new_row = True
            self.parent.thread.append_data = False
            # self.parent.thread.new_row = True
            ## Aqui decir que vas a hacer una nueva fila en la matriz 
            #print("Primera animacion")
        else:
            self.parent.thread.append_data = True
            self.parent.thread.new_row = False
            self.current_animation = 1
            #print("Segunda animacion")
     
            ##### AQUI: Llamar a la función que captura los vectores de vista y los va almacenando
        #print("Cambia")
    
    
    ######## Related to animations
    
    ## Stop initial timer and start calibration timers
    @pyqtSlot()
    def startCalibration(self):
        self.initial_timer.stop()
        #self.parent.setCurrentIndex(3) ## Comentar esto y luego descomentar lo de abaixo
        self.label.setText(" ") # Delete label text
        
        # launch all animations in a sequence of animations
        for i in range(len(positionsList)):
            self.doDotAnimation(positionsList[i][0],positionsList[i][1])

    @pyqtSlot()
    def changeWindow(self):
        # Stop capturing gaze vectors and compute mapping function
        self.parent.thread.append_data = False
        self.parent.thread.new_row = False
        self.parent.thread.calibrate_data = True
        self.parent.showingFrame = True 
        self.parent.setCurrentIndex(3) # De momento que mande a la visualizacion, pero al 2 esta bien por el keyboard

    
    # @pyqtSlot(int, int)
    # def doDotAnimation_simple(self, endPositionX, endPositionY):
        
    #     dotMovingAnimation = QPropertyAnimation(self.calibrationDotBase, b"pos")
    #     dotMovingAnimation.setEndValue(QPoint(endPositionX, endPositionY))
    #     dotMovingAnimation.setDuration(300)
        
        
        
        

    @pyqtSlot(int, int)
    def doDotAnimation(self, endPositionX, endPositionY):
        

        dotMovingAnimation = QPropertyAnimation(self.calibrationDotBase, b"pos")
        dotMovingAnimation.setEndValue(QPoint(endPositionX, endPositionY))
        dotMovingAnimation.setDuration(300)

        animation = QVariantAnimation(
            self.calibrationDotBg,
            valueChanged=self._animateDotCentre,
            startValue=0.85,
            endValue=0.2,
            duration=3000
        )
        
        animation.setDirection(QAbstractAnimation.Forward)
        
        ## Sequential animations
        self.anim_group.addAnimation(dotMovingAnimation)
        self.anim_group.addAnimation(animation)
        self.anim_group.start()
        
        
    def _animateDotCentre(self, value):
        qss = """
        """
        grad = "background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:{value}, fx:0.5, fy:0.5, stop:0 rgba(255, 0, 0, 255), stop:0.519685 rgba(255, 0, 0, 255), stop:0.524752 rgba(255, 255, 255, 0), stop:0.99802 rgba(255, 255, 255, 0));".format(
            value=value
        )
        qss += grad
        self.calibrationDotBg.setStyleSheet(qss)
    
    
        

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
        
    
    def keyPressEvent(self, event):
        if event.key() == qtKeys.Key_Space:
            QApplication.instance().quit()
            
        elif event.key() == qtKeys.Key_A:
            self.thread.getMouseControl = False
    
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