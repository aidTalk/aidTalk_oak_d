#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:37:36 2021

@author: josesolla
"""

import time
import numpy as np
import cv2
from sklearn.svm import SVR
import matplotlib.pyplot as plt

DIST_CALIBR_MESSAGE = 'Place your face inside the ellipse until green!'
DIST_CALIBR_OK_MESSAGE = 'Distance well calibrated!'
DIST_CALIBR_MESSAGE_POS = np.array([1/2, 1/10])

INITIAL_DOT_MESSAGE = 'Look at the dots!' 

# Normalized dot positions for the dot calibration process
### 9-dot option. From GazeRecorder, initial commit
# DOT_LIST = np.array([[1/2, 1/2], ## Initial dot ----- (x,y)
#                       [1/20, 1/16], [1/2, 1/16], [19/20, 1/16], # Calibraiton dots
#                       [1/20, 1/2], [1/2, 1/2], [19/20, 1/2], 
#                       [1/20, 5/6], [1/2, 5/6], [19/20, 5/6]])


## From: Towards accurate eye tracker calibration methods and procedures

#### 11-dot group 13. Se constata cierta mejoría respecto a 9
DOT_LIST = np.array([[1/2, 1/2], ## Initial dot ----- (x,y)
                      [0.05, 0.05], [0.50, 0.95], [0.95, 0.05], # Calibraiton dots
                      [0.50, 0.05], [0.95, 0.95], [0.05, 0.95], 
                      [0.50, 0.50], [0.75, 0.25], [0.25, 0.75], 
                      [0.75, 0.75], [0.25, 0.25]])

# DOT_LIST = np.array([[0, 0],
#  [34, 10],
#  [610, 730],
#  [1186, 10],
#  [610, 10],
#  [1186, 730],
#  [34, 730],
#  [610, 370],
#  [930, 170],
#  [290, 570],
#  [930, 570],
#  [290, 170]])

#### 11-dot group 2. Maybe dots 4 and 6 are not used. No se observa mejoría, empeora incluso
# DOT_LIST = np.array([[1/2, 1/2], ## Initial dot ----- (x,y)
#                       [0.95, 0.50], [0.05, 0.50], [0.50, 0.50], # Calibraiton dots
#                       [0.60, 0.80], [0.40, 0.20], [0.40, 0.80], 
#                       [0.60, 0.20], [0.80, 0.80], [0.20, 0.20], 
#                       [0.80, 0.20], [0.20, 0.80]])




### 13-dot option
# DOT_LIST = np.array([[1/2, 1/2], ## Initial dot ----- (x,y)
#                       [0.05, 0.05], [0.5, 0.95], [0.95, 0.5], # Calibraiton dots
#                       [0.5, 0.05], [0.05, 0.95], [0.05, 0.5], 
#                       [0.95, 0.95], [0.95, 0.05], [0.5, 0.5], 
#                       [0.275, 0.725], [0.725, 0.275], [0.725, 0.725], [0.275, 0.275]])


# DOT_LIST = np.array([[1/2, 1/2], ## Initial dot ----- (x,y)
#                      [1/20, 1/16], [1/2, 1/16], [19/20, 1/16], # Calibraiton dots
#                      [1/20, 1/3], [1/2, 1/3], [19/20, 1/3],
#                      [1/20, 1/2], [1/2, 1/2], [19/20, 1/2], 
#                      [1/20, 5/6], [1/2, 5/6], [19/20, 5/6]])

DOT_SIZE = 20

INITIAL_DOT_TIMER = 6 # 6seconds for the initial dot timer
WAITING_DOT_TIME = 2.5 # 2 seconds
finalStep = 2 # Moving dots and distance steps
OK_DIST_TIMER = 2
GAZE_VECTOR_WAITER = 0.65 # 0.5 seconds of initial wait to stack gaze vectors for each dot position

paintDistCalibration = False

class Calibrator():
    
    """
        Calibrator object of GazeTracker
    """

    
    
    def __init__(self, gazeTracker, windowSize, mapping='poly'):
        
        # Forming GazeTracker object
        self.gazeTracker = gazeTracker
        
        # Size of the frame in the calibration stage
        self.windowSize = windowSize
        
        # Self-control variables
        self.step = 0
        self.initial_timer = 0
        self.dot_step = 0
        self.distance_ok_timer = 0
        
        # Distance calibration control variable
        self.distance_ok = False
        
        # List to append the gaze points (g), appends points per calibration dot a means them to get gazeVector[i]
        self.gaze_leye_list = None 
        self.gaze_reye_list = None 
        self.gazeVectors_leye = None # Matrix of gaze vectors per calibr point (cols - x,y)
        self.gazeVectors_reye = None # Matrix of gaze vectors per calibr point (cols - x,y)
        self.paintDistCalibration = False

        self.mappingFunctionType = mapping
        
        
        ## Control variable to know if it is already calibrated
        self.calibrated = False
        
        # Head pose angles array during calibration
        self.HPAngles = []
        
        self.screenProjectionList = []
    
    def calibrate(self):
        
        """
            Calibrates gaze tracker objects.
            Updates the attributes of the GazeTracker object passed in Calibrator constructor. Creates the mapping function coefficients
        """
        
        # Initialize gaze variables to None
        if self.step==0:
            self.gaze_leye_list = None 
            self.gaze_reye_list = None 
            self.gazeVectors_leye = None # Matrix of gaze vectors per calibr point (cols - x,y)
            self.gazeVectors_reye = None #
            self.step = 2 # Start calibration process
            self.initial_timer = time.time()
            self.calibrated = False
            
        # First calibration step
        elif self.step==1:
            #self.paintDistCalibration = False
            self._calibrateDist()
        
        # Second calibration step
        elif self.step==2:
            #self.paintDistCalibration = False
            self._calibrateDots()
            self.HPAngles.append(self.gazeTracker.pose_angles)
            self.screenProjectionList.append(self.gazeTracker.HeadScreenProjection)
        
        # Calibration done!
        elif self.step>2: 
            
            # Calculate calibration mapping function coefficients!
            self.gazeTracker.calib_coeffs_leye = self.mapGaze(self.gazeVectors_leye)
            self.gazeTracker.calib_coeffs_reye = self.mapGaze(self.gazeVectors_reye)
            
            # Mean pose during calibration and standard deviation
            self.gazeTracker.calibration_pose_angles = np.mean(np.asarray(self.HPAngles), axis=0)
            print(self.gazeTracker.calibration_pose_angles)
            print(np.std(np.asarray(self.HPAngles), axis=0))
            
            ## Mean screen projection during calibration. Igual es bueno filtrar con std
            self.gazeTracker.screenProjectionRef = np.mean(self.screenProjectionList, axis=0)
            
            # Re-set variables
            self.gazeTracker.calibrate = False
            self.step = 0
            self.calibrated = True
            
        
        #return return_frame, self.step
    
    
    def _calibrateDist(self):
        
        """
            Distance camera to user calibration method. 
            Necessary for head rotation correction, work in progress...
        """
        
        # Set frame size and base
        #self.gazeTracker.frame = np.ones(self.windowSize).astype(np.uint8)*60
        
        # Create ellipse mask translucid with border (Esto igual puede hacerse en constructor)
        mask   = np.zeros(self.gazeTracker.frame.shape[0:2]).astype(np.uint8)
        center = (int(self.gazeTracker.frame.shape[1]/2), int(self.gazeTracker.frame.shape[0]/2))
        axis   = (int(2*self.gazeTracker.frame.shape[1]/13), int(self.gazeTracker.frame.shape[1]/5))
        cv2.ellipse(mask,center,axis,0,0,360,255,-1)
        mask[mask==0] = 70
        mask = mask/255
        
        self.gazeTracker.frame[:,:,0] = self.gazeTracker.frame[:,:,0]*mask
        self.gazeTracker.frame[:,:,1] = self.gazeTracker.frame[:,:,1]*mask
        self.gazeTracker.frame[:,:,2] = self.gazeTracker.frame[:,:,2]*mask
        # b_ch, g_ch, r_ch = cv2.split(self.gazeTracker.frame)
        # b_ch, g_ch, r_ch = (b_ch*mask).astype(np.uint8), (g_ch*mask).astype(np.uint8), (r_ch*mask).astype(np.uint8)
        # self.gazeTracker.frame = cv2.merge((b_ch, g_ch, r_ch))
        cv2.ellipse(self.gazeTracker.frame,center,axis,0,0,360,(127,127,127),2)
        
        # Add text when is not well fitted
        if not self.distance_ok:
            cv2.putText(self.gazeTracker.frame, DIST_CALIBR_MESSAGE, (center[0]-600, int(self.gazeTracker.frame.shape[0]/12)), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        
        # If its centered and fits ellipse width
        if self._isCentered(center) and self._hasSameWidth(axis[0]*2):
            
            self.distance_ok = True
            self.distance_ok_timer = time.time()
        
        if self.distance_ok:
            
            # Ellipse border in green
            cv2.ellipse(self.gazeTracker.frame,center,axis,0,0,360,(0,200,0),4)
            
            # Add ok message
            cv2.putText(self.gazeTracker.frame,  DIST_CALIBR_OK_MESSAGE, (center[0]-300, int(self.gazeTracker.frame.shape[0]/12)), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
           
            # Wait 3 seconds
            if (time.time() - self.distance_ok_timer > OK_DIST_TIMER):
                self.distance_ok = False
                self.step = self.step + 1 # Next calibration step
                self.initial_timer = time.time()
                
         
                
        # For debbuging purposes 
        if self.paintDistCalibration:
            cv2.circle(self.gazeTracker.frame, center, 2, (255,0,0), 2)
            cv2.circle(self.gazeTracker.frame, tuple(self.gazeTracker.face_landmarks[0,:]), 2, (0,0,255), 2)
            cv2.circle(self.gazeTracker.frame, tuple(self.gazeTracker.face_landmarks[16,:]), 2, (0,0,255), 2)
            cv2.circle(self.gazeTracker.frame, tuple(self.gazeTracker.face_landmarks[1,:]), 2, (0,0,255), 2)
            cv2.circle(self.gazeTracker.frame, tuple(self.gazeTracker.face_landmarks[15,:]), 2, (0,0,255), 2)
            cv2.circle(self.gazeTracker.frame, tuple(self.gazeTracker.face_landmarks[8,:]), 2, (0,0,255), 2)
       
        
        
    def _hasSameWidth(self, ellipseWidth):
        
        
        sameWidth = False
        faceWidth = abs(self.gazeTracker.face_landmarks[1,0] - self.gazeTracker.face_landmarks[15,0])
        margin = 10
        
        if (faceWidth <= ellipseWidth + margin) and (faceWidth >= ellipseWidth - margin):
            sameWidth = True
        
        return sameWidth
            

    
    def _isCentered(self, ellipseCenter):
        
        centered = False
        
        # Check vertical values
        leftCheekVerticalLimits = np.array([self.gazeTracker.face_landmarks[0,1], self.gazeTracker.face_landmarks[1,1]])
        rightCheekVerticalLimits = np.array([self.gazeTracker.face_landmarks[16,1], self.gazeTracker.face_landmarks[15,1]])
        
        verticalCenter = ellipseCenter[1]
        
        if (verticalCenter>leftCheekVerticalLimits[0] and verticalCenter<leftCheekVerticalLimits[1]):
            
            if (verticalCenter>rightCheekVerticalLimits[0] and verticalCenter<rightCheekVerticalLimits[1]):
         
                # Check horizontal values
                faceHorizontalMiddle = self.gazeTracker.face_landmarks[8,1] # Chin
                margin = 20
                horizontalCenter = ellipseCenter[0]
                
                if (faceHorizontalMiddle>horizontalCenter-margin and faceHorizontalMiddle<horizontalCenter+margin):
                    
                    # Face well centered!
                    centered = True
                    
        
        return centered
    

        
    
    
    def _calibrateDots(self):
        
        """
            Dot calibration method.
            Main calibration process to obtain mapping function coefficients. 
            
        """
        
        
        if self.dot_step == 0: # First dot waits more! Doesn't influence on calibration
            
            
            if (time.time()-self.initial_timer) > INITIAL_DOT_TIMER:
                
                self.initial_timer = time.time()
                self.dot_step = self.dot_step + 1
            
        else:
            
            if (time.time()-self.initial_timer) > GAZE_VECTOR_WAITER:
                
                # Append gaze points to gaze vector
                if self.gaze_leye_list is None:
                    
                    self.gaze_leye_list = self.gazeTracker.gaze_vector_leye
                    self.gaze_reye_list = self.gazeTracker.gaze_vector_reye
                    
                    
                    
                else:
                    
                    self.gaze_leye_list = np.vstack((self.gaze_leye_list, self.gazeTracker.gaze_vector_leye))
                    self.gaze_reye_list = np.vstack((self.gaze_reye_list, self.gazeTracker.gaze_vector_reye))

               
                
            # Next dot
            if (time.time()-self.initial_timer) > WAITING_DOT_TIME:
                
                # Should have detected at least one position, else, do again
                if self.gaze_leye_list is None or self.gaze_reye_list is None:
                    self.initial_timer = time.time()
                    
                else:
                    
                    ## Delete outliers!
                    
                    # Left eye
                    brbr = self.gaze_leye_list
                    mean = np.mean(brbr, axis=0)
                    std = np.std(brbr, axis=0)
                    sup_lim = mean+std
                    inf_lim = mean-std
                    #outliers = np.where(np.logical_and(np.logical_or(brbr[:,0]<inf_lim[0], brbr[:,0]>sup_lim[0]), np.logical_or(brbr[:,1]<inf_lim[1], brbr[:,1]>sup_lim[1])))
                    inliers = np.where(np.logical_or(np.logical_and(brbr[:,0]>inf_lim[0], brbr[:,0]<sup_lim[0]), np.logical_and(brbr[:,1]>inf_lim[1], brbr[:,1]<sup_lim[1])))[0]
                    
                    self.gaze_leye_list = self.gaze_leye_list[inliers,:]
                    
                    # Right eye
                    brbr = self.gaze_reye_list
                    mean = np.mean(brbr, axis=0)
                    std = np.std(brbr, axis=0)
                    sup_lim = mean+std
                    inf_lim = mean-std
                    #outliers = np.where(np.logical_and(np.logical_or(brbr[:,0]<inf_lim[0], brbr[:,0]>sup_lim[0]), np.logical_or(brbr[:,1]<inf_lim[1], brbr[:,1]>sup_lim[1])))
                    inliers = np.where(np.logical_or(np.logical_and(brbr[:,0]>inf_lim[0], brbr[:,0]<sup_lim[0]), np.logical_and(brbr[:,1]>inf_lim[1], brbr[:,1]<sup_lim[1])))[0]
                    
                    self.gaze_reye_list = self.gaze_reye_list[inliers,:]
                    
                    # Mean and add to gazeVector
                    if self.gazeVectors_leye is None:
    
                        self.gazeVectors_leye = np.mean(self.gaze_leye_list, axis=0)
                        self.gazeVectors_reye = np.mean(self.gaze_reye_list, axis=0)
                        

                    else:
                        
                        self.gazeVectors_leye = np.vstack((self.gazeVectors_leye, np.mean(self.gaze_leye_list, axis=0)))
                        self.gazeVectors_reye = np.vstack((self.gazeVectors_reye, np.mean(self.gaze_reye_list, axis=0)))
                        
                    #print(self.gaze_leye_list)
                    #print(self.gazeVectors_leye)
                        
                    # Re-set and next dot
                    self.gaze_leye_list = None
                    self.gaze_reye_list = None
                    self.initial_timer = time.time()
                    self.dot_step = self.dot_step + 1
                

        
        # Stop dot calibration
        if self.dot_step == len(DOT_LIST):
            
            self.dot_step = 0
            self.initial_timer = 0
            self.step = self.step+1 # Next step
            
            return
        
        
        # Base frame
        self.gazeTracker.frame = np.ones((self.windowSize[0], self.windowSize[1], 3)).astype(np.uint8)*60
        
        # Draw dots
        dot_position = self._getDotPosition()
        cv2.circle(self.gazeTracker.frame, dot_position, DOT_SIZE, (71, 99, 255), -1)
        cv2.circle(self.gazeTracker.frame, dot_position, DOT_SIZE, (5, 5, 160), 3)
        cv2.circle(self.gazeTracker.frame, dot_position, 5, (0, 0, 0), -1)
        cv2.circle(self.gazeTracker.frame, dot_position, DOT_SIZE, (0,0,0), 1)
        
        # Draw cross on dot
        cv2.line(self.gazeTracker.frame, (int(dot_position[0]-DOT_SIZE-10), dot_position[1]), (int(dot_position[0]+DOT_SIZE+10), dot_position[1]), (0,0,0), 1)
        cv2.line(self.gazeTracker.frame, (dot_position[0], int(dot_position[1]-DOT_SIZE-10)), (dot_position[0], int(dot_position[1]+DOT_SIZE+10)), (0,0,0), 1)
        
        # Will also write the text
        if self.dot_step == 0: 
            cv2.putText(self.gazeTracker.frame, INITIAL_DOT_MESSAGE, (dot_position[0]-200, dot_position[1]-150), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 2, cv2.LINE_AA)
        
            

    def _getDotPosition(self):
        
        """
            Gets dot position in the screen according to the current dot in DOT_LIST
        """
        
        position = DOT_LIST[self.dot_step, :].copy()
        position[0] = position[0]*self.windowSize[1]
        position[1] = position[1]*self.windowSize[0]
        #position = np.flip(position)
        
        return tuple(map(int, position))
        
    
    def filterGazeData(self, gazeVectorsMatrix):
        
        pass
    
    def mapGaze(self, gazeVectors):
        
        if self.mappingFunctionType=='poly':
            return self._polyMap(gazeVectors)
        elif self.mappingFunctionType=='SVR':
            return self._SVRMap(gazeVectors)

        
        
    def _polyMap(self, gazeVectors):

        """
            Maps gaze with quadratic function in the dot calibration process:
            ux = a0 + a1gx + a2gy + a3gxgy + a4g2x+ a5g2y
            uy = b0 + b1gx + b2gy + b3gxgy + b4g2x+ b5g2y
            
            Reference:
            
                Y. Cheung and Q. Peng, "Eye Gaze Tracking With a Web Camera in a Desktop Environment," 
                in IEEE Transactions on Human-Machine Systems, vol. 45, no. 4, pp. 419-430, Aug. 2015, 
                doi: 10.1109/THMS.2015.2400442.

        """
        
        # Get gaze vector values of each coordinate
        gx = gazeVectors[:,0]
        gy = gazeVectors[:,1]
        first_col = np.ones((len(gazeVectors),))
        
        
        # Solve first for the x coordinate coefficient values
        # Create coeff-side matrix
        Ax = np.vstack((first_col, gx, gy, gx*gy, gx**2, gy**2)).T
        
        # Ordinate matrix
        Bx = DOT_LIST[1:,0] # x coordinates of dot points. Igual hay que multiplicar por los pixeles en x??
        
        # Solve system
        aCoeffs,_,_,_ = np.linalg.lstsq(Ax, Bx, rcond=None)
        
        # Solve for the y coordinate coefficient values
        
        # Ordinate matrix
        Bx = DOT_LIST[1:,1] # x coordinates of dot points
        
        # Solve system
        bCoeffs,_,_,_ = np.linalg.lstsq(Ax, Bx, rcond=None)
        
        # System coefficients (a and b)
        coeffs = np.vstack((aCoeffs, bCoeffs))
        
        return coeffs

    def _SVRMap(self, gazeVectors):

        """
            Maps gaze with SVR function
            
            Reference:
            
                Y. Cheung and Q. Peng, "Eye Gaze Tracking With a Web Camera in a Desktop Environment," 
                in IEEE Transactions on Human-Machine Systems, vol. 45, no. 4, pp. 419-430, Aug. 2015, 
                doi: 10.1109/THMS.2015.2400442.

        """
        
        ##### Initial commit, one SVR per each component (x or y)
        
        # Get gaze vector values of each coordinate. Training data
        gx = gazeVectors[:,0]
        gy = gazeVectors[:,1]

        # Real dot values shown in screen. Target values
        dotsx = DOT_LIST[1:,0]
        dotsy = DOT_LIST[1:,1]
        
        # SVR
        svrX = SVR(kernel='rbf', C=10, gamma=8)
        svrY = SVR(kernel='rbf', C=10, gamma=8)
        
        # Fit training data with target values
        svrX = svrX.fit(gx.reshape(-1, 1), dotsx)
        svrY = svrX.fit(gy.reshape(-1, 1), dotsy)
        
        # Self training data prediction
        y_svrX = svrX.predict(gx)
        y_svrY = svrY.predict(gy)
        
        lw = 2
        plt.scatter(gx, dotsx, color='darkorange', label='data')
        #plt.hold('on')
        plt.plot(gx, y_svrX, color='navy', lw=lw, label='RBF model')
        #plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
        #plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
        plt.xlabel('data')
        plt.ylabel('target')
        plt.title('Support Vector Regression')
        plt.legend()
        plt.show()

        
       
        
        return None

    def _moveDot(self, initial, final):
        
        """
            Moves dot from initial to final position in the specified time.
            
            Work in progress, not avaliable yet
        """
        pass
    
   