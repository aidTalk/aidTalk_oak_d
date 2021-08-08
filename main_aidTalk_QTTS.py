#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:31:25 2021

@author: josesolla
"""



from PySide2.QtWidgets import QApplication, QDialog, QStackedWidget, QSizePolicy, QDesktopWidget
from PySide2.QtGui import QImage, QPixmap, QCursor, QGuiApplication, QWindow
from PySide2.QtCore import QPropertyAnimation, QPoint, Qt, QTimer, QCoreApplication

from PySide2.QtCore import Signal, Slot, QThread
from PySide2.QtTextToSpeech import QTextToSpeech


import sys
import cv2
import numpy as np
import argparse
import time


## UIs
from user_interfaces import Ui_frameShowing, Ui_calibrationWindow, Ui_menuWindow, Ui_writingWindow

# Bring Gaze Tracker object!
from gaze_tracker import GazeTracker

# import tkinter as tk

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FOCAL_LENGTH_CM = 0.63
FOCAL_LENGTH_PX = 1085.54
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
        
        self.window = QWindow()
        
        self.window.setScreen(self.screen)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(WEBCAM_SOURCE)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
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
                                #QCursor.setPos(self.screen, int(mouseX), int(mouseY))
                                self.moveCursor(int(mouseX), int(mouseY))
                                #QCursor.setPos(self.screen, QPoint(int(mouseX), int(mouseY)))

                    except Exception as err:
                        print(str(err))
                        # print("hubo excepcion")
                        pass
                
                
                self.accumulated_time =  self.accumulated_time + (time.time()-start)
                self.accumulated_cont = self.accumulated_cont+1
                if self.show_image: self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()
        print("AVERAGE Processing time: %0.3f" % (self.accumulated_time/self.accumulated_cont))
        print("AVERAGE FPS: %d" % (self.accumulated_cont/self.accumulated_time))
    
    
    def moveCursor(self, x, y):
        
        sscreen = QGuiApplication.primaryScreen()
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
        self.parent.setCurrentIndex(1)
        time = self.parent.calibrationWindow.initial_waiting_time
        self.parent.calibrationWindow.initial_timer.start(time)
        
    @Slot()
    def goToHead(self):
        self.parent.thread.getMouseControl = True
        self.parent.thread.gazeTracker.HeadControlled()
        self.parent.thread.isGazedControlled = False
        self.parent.setCurrentIndex(2)
        self.parent.thread.show_image = True

    
    


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
        self.ui.label.setText("Mire fijamente al centro de los puntos")
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
        self.parent.setCurrentIndex(2) # De momento que mande a la visualizacion, pero al 2 esta bien por el keyboard
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
        self.parent.setCurrentIndex(2)
    
    
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
                    ['m','n','o'],
                    ['p','q','r','s'],
                    ['t','u','v'],
                    ['w','x','y','z'],
                    [' ','?','!', ',']]
        
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
        self.ui.pushButton12.clicked.connect(self.parent.closeApp)
        self.ui.pushButton13.clicked.connect(self.b13_clicked)
    
        
    
    @Slot()    
    def write(self):
        
        ###writing intreface control  
        #print(self.timer.remainingTime())
        if self.key_index is not None:
            if self.key_index == 5 or self.key_index == 7 or self.key_index == 8:
                self.robin_index%=4
            
            else:
                self.robin_index%=3
                    
            self.text2speech = self.text2speech + self.dic[self.key_index][self.robin_index]
            #print(self.text2speech)
            
            self.ui.textEdit.setText(self.text2speech)
            self.robin_index = -1    
            self.timer.stop()
        
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
        if self.timer.remainingTime()>0and self.key_index!=6:
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
        
    @Slot()    
    def b10_clicked(self):
        """
        Under construction, to send self.thread.text2speech to the tts

        """
        self.qtts.say(self.text2speech)
        self.text2speech = ''
        pass
    
    @Slot()
    def b11_clicked(self):
        
        self.text2speech = self.text2speech[:-1]
        self.ui.textEdit.setText(self.text2speech)
        #print(self.text2speech)
       
    # @pyqtSlot()
    # def b12_clicked(self):
    #     pass
    
    @Slot()
    def b13_clicked(self):
        self.parent.setCurrentIndex(0)
    
    



class aidTalkApp(QStackedWidget):   
    
    def __init__(self, selectedScreenObject):
        super(aidTalkApp, self).__init__()
        
        # Screen options
        self.selected_screen = MONITOR_SOURCE
        #print(selectedScreenObject.size().width(), selectedScreenObject.size().height())
        self.screen = selectedScreenObject
        print(self.screen)
        # Middel psotion of the screen widget
        self.middle = ((DOT_LIST[0,:]*[self.screen.size().width(), 
                                       self.screen.size().height()]) - 30).astype(int).tolist()
        

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
            self.closeApp()
            
        elif event.key() == Qt.Key_A:
            self.thread.getMouseControl = False
            
            
        elif event.key() == Qt.Key_Up: # Re-calibrate
        
            if self.thread.isGazedControlled:
                self.setCurrentIndex(1)
                self.thread.getMouseControl = False
                self.calibrationWindow.ui.calibrationDotBase.move(self.middle[0], self.middle[1])
                self.calibrationWindow.ui.label.setText("Mire fijamente al centro de los puntos")
                time = self.calibrationWindow.initial_waiting_time
                self.calibrationWindow.initial_timer.start(time)
            
        elif event.key() == Qt.Key_Down:
            self.closeApp()
           
        # For changing to visualization mode
        elif event.key() == Qt.Key_V:
            self.setCurrentIndex(3)
            self.thread.show_image = True
            self.thread.getMouseControl = False
            
        elif event.key() == Qt.Key_M:
            self.setCurrentIndex(0)
            self.thread.getMouseControl = False
            
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

