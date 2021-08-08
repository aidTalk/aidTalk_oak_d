#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:14:16 2021

@author: josesolla
"""

#import time
import cv2
import collections
import numpy as np
from calibrator_v2 import Calibrator
from detector import Detector
from scipy.stats import norm

import pickle
import warnings

# CPU inference
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#from models.elg_keras import KerasELG
#from keras import backend as K

# import onnxruntime
# import time

# Control variables
EYE_AR_THRESH = 0.2
AVERAGE_FACE_BREADTH = 130 #mm

## gaze smoothing
smoothing_window_size = 4
smoothing_coefficient_decay = 0.5
smoothing_coefficients = None

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


class GazeTracker(object):
    
    """
    GazeTracker(frameShape, [shown_image, [focal_length]])
    
    Class to instantiate GazeTracker objects
    
    NOTE: It's important to set the constructor with these arguments in the following order.
        1. Frame shape: tuple, list, ndarray
            Frame capture shape (height, width)
            
        2. (Optional) Shown Image: OpenCV image, image path
            Image to be shown after the calibration process in the gaze tracking state.
            If not given will show capture frame.
            
        3. (Optional, Not Working Yet) Aparent focal camera length in pixels: int, float
            Aparent focal camera length in pixels used for the head rotation correction.
            Not working yet, it's optional and won't be used in the program
    ...

    Attributes
    ----------

    Methods
    -------
    Update(frame)
        GazeTracker update method. Depending on the current state performs detection, calibration, tracking, gaze tracking or blinking detections.
    
    Calibrate()
        Sets calibration state
        
    CheckBlinkings()
        Sets blinking checker
    """
    
    def __init__(self, screen_size = (1920, 1080), camera_resolution = (1280, 720), focal_length_cm = 0.63, focal_length_px = 1085.54, mappingFunct = 'poly', loadCalibration=False):
        
        """
        Parameters
        ----------
        screen_size : tuple, list, ndarray of int
            Screen monitor resolution (width, height)
        
        focal_length : number
            Aparent focal length of the camera in pixels. 
        
        
        """
        
        if isinstance(screen_size, (tuple)):
            pass
        elif isinstance(screen_size, (list, np.ndarray)):
            screen_size = tuple(screen_size)
        else:
            screen_size = (1920, 1080)
            warnings.warn("Warning: screen_size must be "+
                      "horizontal and vertical resolutions of the screen, "+
                      "as a tuple, list or np.ndarray of integers. "+
                      "Default 1920x1080 px has been set, but this may cause "+
                      "the application to function incorrectly")
        
        # Screen size attribute
        self.screen_size = screen_size
        
        
        if isinstance(camera_resolution, (tuple)):
            pass
        elif isinstance(camera_resolution, (list, np.ndarray)):
            camera_resolution = tuple(camera_resolution)
        else:
            camera_resolution = (1280, 720)
            warnings.warn("Warning: camera_resolution must be "+
                          "horizontal and vertical resolution of webcam capture, "+
                          "as a tuple, list or np.ndarray of integers. "+
                          "Default 1280x720 px has been set, but this may cause "+
                          "the application to function incorrectly")
        
        # Camera resolution attribute
        self.camera_resolution = camera_resolution
        
        # Focal length attributes
        self.focal_length_cm = focal_length_cm
        self.focal_length_px = focal_length_px
        
        # Mapping function type attribute
        self.mapping_function = mappingFunct
        
        
        ## ---------------------------------------------------------------- ##
        # Forming images
        self.frame = None
        self.left_eye = None
        self.right_eye = None
        
        # Control variables
        self.detected = False
        self.calibrate = False
        
        # Forming objects: detector and calibrator
        if self.mapping_function == 'poly':
        	self.calibrator = Calibrator(self, self.screen_size)
        elif self.mapping_function == 'SVR':
        	self.calibrator = Calibrator(self, self.screen_size, mapping=self.mapping_function)


        self.detector = Detector(self, detect_conf=0.5, track_conf=0.5)
        
        # Forming gaze objects, pupil centers within eye frames
        self.lPupil_center = None
        self.rPupil_center = None
        self.gaze_vector_leye = None
        self.gaze_vector_reye = None
        self.face_landmarks = None
        
        ## Control variables of where the eye frame starts
        self.leye_corner = None
        self.reye_corner = None
        
        # Calibration coefficients for the mapping function
        self.calib_coeffs_leye = None
        self.calib_coeffs_reye = None
        
        # Rotation matrix for the head pose estimation
        self.rotation_matrix = None
        self.distance_user_screen = 1000
        

        ### Gaze entries window variable for gaze smoothing
        self.gaze_entries_window = collections.deque(maxlen=smoothing_window_size)
        
        ### Head Pose projection entries window variable for gaze smoothing
        self.head_entries_window = collections.deque(maxlen=smoothing_window_size)
        
        
        ## ---------------------------------------------------------------- ##
        
        ### Head Pose Related
        self.pose_angles = None
        self.calibration_pose_angles = None
        
        
        ### Head Distance related
        self.reference_head_width_cm = 13
        self.reference_head_heigth_cm = 15.5
        self.nose_to_temple_correction = 3.5
        self.cm2px = self.focal_length_px / self.focal_length_cm
        self.px2cm = self.focal_length_cm / self.focal_length_px
        
        # Camera intrinsic parameters
        self.camera_matrix = np.array(
            [[self.camera_resolution[0] , 2                 ,         int(self.camera_resolution[0]/2)],
             [0                         , self.camera_resolution[1] , int(self.camera_resolution[1]/2)],
             [0                         , 0                 , 1  ]], dtype = "double"
        )
        # Rotation Matrix
        self.R = None
        # Translation Vector
        self.t = None
        # Pixel distance from webcam to head
        self.distance_px = None
        
        
        
        ## ---------------------------------------------------------------- ##
        
        ## Screen projection from head pose
        self.HeadScreenProjection = None
        
        ### Gaze screen projection
        self.GazeScreenProjection = (None, None)
        
        ### Control for gaze output (gaze controlled or head controlled)
        self.control_with_gaze = False
        self.control_with_head = False
        
        
        #### Timers attributes
        # self.accum_detection = 0
        # self.accum_gazeVector = 0
        # self.accum_headPose = 0
        # self.accum_resize = 0

        # Load if there is a calibration database
        if loadCalibration:
            try:
                self._loadCalibration()
                self.calibrator.calibrated = True # Tell calibrator that it was already calibrated
            except Exception:
                self.calibrator.calibrated = False
                warnings.warn("Warning: No calibration data found, system will be recalibrated.")
                pass
        
        
        
    def Update(self, frame):
        
        """
            GazeTracker update method.
            Depending on the current state performs detection, calibration, tracking, gaze tracking or blinking detections.
            
            frame: OpenCV image
            
            returns:
                OpenCV image processed
        """
        
        if frame is None:
            warnings.warn("Warning: Frame is none, returning None.")
            return None
        
        self.frame = frame
        
        
        
        # start = time.time()
        self.detector.detect(show=False)
        # self.accum_detection = self.accum_detection + (time.time()-start)
        
        # start = time.time()
        self._getGazeVector(show=False)
        # self.accum_gazeVector = self.accum_gazeVector + (time.time()-start)
        
        # start = time.time()
        self._getHeadRotation(show=False)
        # self.accum_headPose = self.accum_headPose + (time.time()-start)
        #self._getHeadDistance()
        
        #self._getRotationCorrection(1000, show=True)
        #self._estimateGazeAndProjection_v2(show=True)
        #self.getEyesRotations()


        # Calibrate if asked
        if self.calibrate:
            
            self.calibrator.calibrate()
            self.frame = cv2.resize(self.frame, (self.screen_size[0], self.screen_size[1]))
            # Check if calibration is done
            if self.calibrator.calibrated:
                # Save calibration
                self._saveCalibration()
        
        # Once the calibration process is done and we have our mapping function coefficients
        if self.calibrator.calibrated: # It was calibrated!
            
            # Get screen position according to gaze
            u = self._trackScreenPosition(show=True)
            self.GazeScreenProjection = u


            
        # Return frame after modifications
        # start = time.time()
        #self.frame = cv2.resize(self.frame, (self.focal_width, self.focal_height))
        # self.accum_resize = self.accum_resize + (time.time()-start)
        
        
        ### Return accordingly to the control selected
        if self.control_with_head:
            return self.frame, self.HeadScreenProjection[0], self.HeadScreenProjection[1]
        elif self.control_with_gaze:
            return self.frame, self.GazeScreenProjection[0], self.GazeScreenProjection[1]
        else:
            return self.frame, None, None


    
    def HeadControlled(self):
        self.control_with_head = True
        self.control_with_gaze = False
        #pass
    
    def GazeControlled(self):
        self.control_with_gaze = True
        self.control_with_head = False
        #pass
    
    def Close(self):
        
        self.detector.face_mesh.close()
        del self.detector.sess
        
    def printInfo(self):
        print(self.pose_angles)
    
    def Calibrate(self):
        
        """
            Set calibration state
        """
        
        self.calibrate = True
        # re-set calibration coefficients
        self.calib_coeffs_leye = None
        self.calib_coeffs_reye = None
    
    def isCalibrationFinished(self):
        
        if self.calibrator.calibrated:
            return True
        else:
            return False
    

    
    ###### PRIVATE METHODS FOR OPERATIONS ######
   
    
    ######## GAZE ESTIMATION ########
    
    def _getGazeVector(self, show=False, scale = 1): 
        
        """
            Computes gaze vector as the difference of pupil center and inner eye
            
            scale: scales gaze vector to gain more precission if needed
        """
        
        
        if self.face_landmarks is not None:
            # Forming positions for vector mapping
            inner_leye = self.face_landmarks[133,:].copy()
            inner_reye = self.face_landmarks[362,:].copy()
            pupil_leye = self.lPupil_center
            pupil_reye = self.rPupil_center
            
                
            if (inner_leye is not None and inner_reye is not None) and (pupil_leye is not None and pupil_reye is not None) and (self.leye_corner is not None and self.reye_corner is not None):
            
                # Get adjusted inner eyes to eye frames
                
                # Normalize inner eyes to eye frame upleft corner reference
               
                inner_leye = inner_leye-self.leye_corner
                inner_reye = inner_reye-self.reye_corner
                
                if show:
                    cv2.circle(self.left_eye, tuple(map(int, inner_leye)), 2, (0, 0, 255), thickness=1)
                    cv2.circle(self.right_eye, tuple(map(int, inner_reye)), 2, (0, 0, 255), thickness=1)
                    
                    cv2.circle(self.left_eye, tuple(map(int, pupil_leye)), 2, (0, 255, 0), thickness=1)
                    cv2.circle(self.right_eye, tuple(map(int, pupil_reye)), 2, (0, 255, 0), thickness=1)
                
                # Upscale
                if scale !=1:
                    inner_leye = inner_leye*scale
                    inner_reye = inner_reye*scale
                    pupil_leye = pupil_leye*scale
                    pupil_reye = pupil_reye*scale
                    
    
                # Get gaze vectors
                self.gaze_vector_leye = inner_leye - pupil_leye
                self.gaze_vector_reye = inner_reye - pupil_reye

  
    def _trackScreenPosition(self, show=False):
        
        """
            Solve mapping function quadratic polynomial 2-D equation for each eye:
            ux = a0 + a1gx + a2gy + a3gxgy + a4g2x+ a5g2y
            uy = b0 + b1gx + b2gy + b3gxgy + b4g2x+ b5g2y

        """
        # if self.shown_img is not None:
        #     self.frame = self.shown_img.copy().astype(np.uint8)
        
        # Gaze vector positions
        gx_l, gy_l = self.gaze_vector_leye
        gx_r, gy_r = self.gaze_vector_reye
        
        ## Y hacer aqui la corrección?
        
        
        ux_l = int(self.screen_size[0]*(self.calib_coeffs_leye[0,0] + self.calib_coeffs_leye[0,1]*gx_l + 
                   self.calib_coeffs_leye[0,2]*gy_l + self.calib_coeffs_leye[0,3]*gx_l*gy_l + 
                   self.calib_coeffs_leye[0,4]*gx_l**2 + self.calib_coeffs_leye[0,5]*gy_l**2))
        
        uy_l = int(self.screen_size[1]*(self.calib_coeffs_leye[1,0] + self.calib_coeffs_leye[1,1]*gx_l + 
                   self.calib_coeffs_leye[1,2]*gy_l + self.calib_coeffs_leye[1,3]*gx_l*gy_l + 
                   self.calib_coeffs_leye[1,4]*gx_l**2 + self.calib_coeffs_leye[1,5]*gy_l**2))
        
        ux_r = int(self.screen_size[0]*(self.calib_coeffs_reye[0,0] + self.calib_coeffs_reye[0,1]*gx_r + 
                   self.calib_coeffs_reye[0,2]*gy_r + self.calib_coeffs_reye[0,3]*gx_r*gy_r + 
                   self.calib_coeffs_reye[0,4]*gx_r**2 + self.calib_coeffs_reye[0,5]*gy_r**2))
        
        uy_r = int(self.screen_size[1]*(self.calib_coeffs_reye[1,0] + self.calib_coeffs_reye[1,1]*gx_r + 
                   self.calib_coeffs_reye[1,2]*gy_r + self.calib_coeffs_reye[1,3]*gx_r*gy_r + 
                   self.calib_coeffs_reye[1,4]*gx_r**2 + self.calib_coeffs_reye[1,5]*gy_r**2))
        
        
        
        # Mean both eye positions
        mean_ux = int((ux_l+ux_r)/2)
        mean_uy = int((uy_l+uy_r)/2)
        
        
        if mean_ux<0:
            mean_ux = 0
        elif mean_ux>self.screen_size[0]:
            mean_ux = self.screen_size[0]
        
        if mean_uy<0:
            mean_uy = 0
        elif mean_uy>self.screen_size[1]:
            mean_uy = self.screen_size[1]
        
       
        # Smooth gaze commit
        u = (mean_ux, mean_uy)
        self.gaze_entries_window.append(u)
        self._filter_gaussian_gaze(self.gaze_entries_window)
        # self._smooth_gaze(smoothing_coefficients, self.gaze_entries_window)
        u = self.gaze_entries_window[-1]

        
        # Create mask
        # mask = np.ones(self.frame.shape[0:2])*0.6
        # cv2.circle(mask, u, 40, 0.95, -1)
        # cv2.circle(mask, u, 40, 1, 2)
        
        # self.frame[:,:,0] = self.frame[:,:,0]*mask
        # self.frame[:,:,1] = self.frame[:,:,1]*mask
        # self.frame[:,:,2] = self.frame[:,:,2]*mask
        
        if show: 
            cv2.circle(self.frame, u, 10, (0, 215, 255), -1)


        return u
    
    #################################################
    # HACE LA MEDIA DE N ESTIMACIONES PREVIAS XDDDD #
    # np.mean(gaze_history, axis=0)
    #################################################
    
    def _filter_gaussian_gaze(self, gaze_entries_window, meaniness=0.99):
        
        """
            Meaniness: The more 9 decimals added the more it will behave as a mean filter.
                0.9999999999999999 is the limit
                
                0.99 esta guay para ventana de 7. PERO MIL VECES MEJOR CON VENTANA DE 4
        """
        a = len(gaze_entries_window)
        
        if a < smoothing_window_size:
            """If not enough frames for smoothing window."""
            return
        
        if np.any(gaze_entries_window == None):
            """Any frame has zero faces detected."""
            return 
        
        x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), smoothing_window_size)
        coefficients = norm.cdf(x, loc=0, scale=1)
        coefficients = coefficients/np.sum(coefficients)
        
        gauss = tuple(map(int, np.sum(np.array([coefficients]).T*np.asarray(gaze_entries_window), axis=0)))
        gaze_entries_window[-1] = gauss
         
    
    def _mean_gaze(self, gaze_entries_window):
        #print(gaze_entries_window)
        a = len(gaze_entries_window)
        
        if a < smoothing_window_size:
            """If not enough frames for smoothing window."""
            return

        if np.any(gaze_entries_window == None):
            """Any frame has zero faces detected."""
            return 
        
        media = tuple(np.mean(np.asarray(gaze_entries_window), axis=0).astype(int))
        gaze_entries_window[-1] = media
        
    
    ###### VOLVER A PROBAR ESTO: Permitiría más muestras pasadas para estimar pero teniendo mayor peso en las muestras más recientes, y es bien sencillo fijo
    # np.sum(lista_np*smoothing_coefficients, axis=0) # lista_np seria gaze entries en numpy
    
    def _smooth_gaze(self, smoothing_coefficients, gaze_entries_window):
        """If there are previous landmark detections, try to smooth current prediction."""
        # Cache coefficients based on defined sliding window size
        if smoothing_coefficients is None:
            coefficients = np.power(smoothing_coefficient_decay,
                                    list(reversed(list(range(smoothing_window_size)))))
            coefficients /= np.sum(coefficients)
            smoothing_coefficients = coefficients.reshape(-1, 1)
    
        # Get a window of frames
        #current_index = _indices.index(frame['frame_index'])
        a = len(gaze_entries_window)
        if a < smoothing_window_size:
            """If not enough frames for smoothing window."""
            return
        # window_indices = _indices[a:current_index + 1]
        # window_frames = [frames[idx] for idx in window_indices]
        # window_num_landmark_entries = np.array([len(f['landmarks']) for f in window_frames])
        if np.any(gaze_entries_window == None):
            """Any frame has zero faces detected."""
            return 
    
        # Apply coefficients to landmarks in window
        window_gazes = np.asarray(gaze_entries_window)
        smoothed_gaze = np.sum(
            np.multiply(window_gazes.reshape(smoothing_window_size, -1),
                        smoothing_coefficients),
            axis=0,
        ).reshape(len(gaze_entries_window[-1]), -1)
        
        gaze_entries_window[-1] = tuple(smoothed_gaze[:,0].astype(int))
        
    
    
    ###### HEAD ROTATION CORRECTION ######
  
    def _getHeadRotation(self, show=False):
        
        if self.face_landmarks is not None:
            ## Get Yaw
            line_34_264 = [self.face_landmarks[34,0], self.face_landmarks[34,1], self.face_landmarks[264,0], self.face_landmarks[264,1]]
            crossover6 = self._point_line(self.face_landmarks[6,:], line_34_264)
            yaw_mean = np.linalg.norm(self.face_landmarks[34,:]-self.face_landmarks[264,:])/2
            yaw_right = np.linalg.norm(self.face_landmarks[34,:]-crossover6)
            yaw = (yaw_mean - yaw_right) / yaw_mean
            yaw = int(yaw * 71.58 + 0.7037)
            
            
            ## Get Pitch
            pitch_dis = -np.linalg.norm(self.face_landmarks[6,:]-crossover6)
            if self.face_landmarks[6,1] < crossover6[1]:
                pitch_dis = -pitch_dis
            pitch = int(1.497 * pitch_dis - 11.97)
            
            
            ## Get Roll
            if abs(self.face_landmarks[33,0] - self.face_landmarks[263,0]) == 0:
                roll_tan = abs(self.face_landmarks[33,1] - self.face_landmarks[263,1])
            else:
                roll_tan = abs(self.face_landmarks[33,1] - self.face_landmarks[263,1]) / abs(self.face_landmarks[33,0] - self.face_landmarks[263,0])
            roll = np.arctan(roll_tan)
            roll = 180*roll/np.pi
            if self.face_landmarks[33,1] > self.face_landmarks[263,1]:
                roll = -roll
            roll = int(roll)
            
            self.yaw = yaw
            #theta = [yaw, pitch, roll]
            theta = [pitch, -yaw, roll]
            self.pose_angles = theta
            
            ### Calculate totation matrix R
            R_x = np.array([[1,         0,                0   ],
                [0,         np.cos(np.deg2rad(theta[0])), -np.sin(np.deg2rad(theta[0])) ],
                [0,         np.sin(np.deg2rad(theta[0])),   np.cos(np.deg2rad(theta[0]))  ]])
             
            R_y = np.array([[np.cos(np.deg2rad(theta[1])),    0,      np.sin(np.deg2rad(theta[1]))  ],
                            [0,                               1,      0                             ],
                            [-np.sin(np.deg2rad(theta[1])),   0,      np.cos(np.deg2rad(theta[1]))  ]
                            ])
                        
            R_z = np.array([[np.cos(np.deg2rad(theta[2])),    -np.sin(np.deg2rad(theta[2])),    0],
                            [np.sin(np.deg2rad(theta[2])),    np.cos(np.deg2rad(theta[2])),     0],
                            [0,                               0,                                1]
                            ])

            R = np.dot(R_z, np.dot( R_y, R_x ))
            
            ##Get head distance
            head_distance_cm = self._getHeadDistance(yaw, pitch)
            
            
            ###Xc=R·Xw+t
            nosetip = self.face_landmarks[1,:]
            nosetip_z = head_distance_cm*self.cm2px
            
            
            Xw = np.array([(nosetip[0] * nosetip_z / self.focal_length_px),
                  (nosetip[1]  * nosetip_z / self.focal_length_px),
                  nosetip_z ])
            t = np.array([((- self.camera_resolution[0] / 2) * nosetip_z / self.focal_length_px),
                  ((- self.camera_resolution[1] / 2) * nosetip_z / self.focal_length_px),
                  0 ])
            
            Xc=np.dot(R,Xw)+t
            
            ### Webcam sensor proyection
            UVWp=np.dot(self.camera_matrix,Xc)
            
            U=UVWp[0]/UVWp[2]
            V=UVWp[1]/UVWp[2]
            
            ## Adjust projection to screen
            U = int(U*self.screen_size[0]/self.camera_resolution[0])
            V = int(V*self.screen_size[1]/self.camera_resolution[1])
            
            ### Do smoothing with previous projections
            projected = (U, V)
            self.head_entries_window.append(projected)
            self._filter_gaussian_gaze(self.head_entries_window)
            # self._smooth_gaze(smoothing_coefficients, self.gaze_entries_window)
            U,V = self.head_entries_window[-1]
            

            if U<0:
                U = 0
            elif U>self.screen_size[0]:
                U= self.screen_size[0]
        
            if V<0:
                V = 0
            elif V>self.screen_size[1]:
                V = self.screen_size[1]
            
            
            self.HeadScreenProjection = (U,V)
            
            if show: cv2.circle(self.frame, (int(U), int(V)), 1, (0,255,255), thickness=3)
            
            
            
            #Showing Arrow
            if show:
                pitchyaw = np.array([pitch, yaw])*np.pi/180
                nosetip = self.face_landmarks[1,:]
                nosetip[0] = int(nosetip[0])
                nosetip[1] = int(nosetip[1])
                
                length = 100.0
                thickness = 2
                color = (255, 0, 0)
                dx = -length * np.sin(pitchyaw[1])
                dy = -length * np.sin(pitchyaw[0])
                cv2.arrowedLine(self.frame, tuple(np.round(nosetip).astype(np.int32)),
                               tuple(np.round([nosetip[0] + dx, nosetip[1] + dy]).astype(int)), color,
                               thickness, cv2.LINE_AA, tipLength=0.2)
                
                cv2.putText(self.frame,f"Head_Yaw(degree): {yaw}",(30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
                cv2.putText(self.frame,f"Head_Pitch(degree): {pitch}",(30,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
                cv2.putText(self.frame,f"Head_Roll(degree): {roll}",(30,200),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
            
        #return R, (U,V)
    
    
    def _point_line(self, point,line):
        x1 = line[0]  
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
    
        x3 = point[0]
        y3 = point[1]
        if (x2-x1) == 0:
            k1 = (y2 - y1)*1.0 /np.finfo(float).eps
        else:
            k1 = (y2 - y1)*1.0 /(x2 -x1) 
        b1 = y1 *1.0 - x1 *k1 *1.0
        if k1 == 0:
            k1 = np.finfo(float).eps
           
        k2 = -1.0/k1
        b2 = y3 *1.0 -x3 * k2 *1.0
        x = (b2 - b1) * 1.0 /(k1 - k2)
        y = k1 * x *1.0 +b1 *1.0
        return np.array([x,y])
    
    def _getHeadDistance(self, yaw, pitch):
        
        """
        Focal distance dependencies
        focal_length = reference_head_heigth_cm * frame_width_px / Sensor_width_cm
        
        Surface Pro data: -F = 3mm
                          -Sensor_width = 1/3" 
        
        Correction: difference in Z axis from nose to temple
        
        reference_head_widtreference_head_heigth_cm: real head width within temples

        """
        #Mean of two X and Y vectors
        head_width_proyection_px = np.linalg.norm(
            self.face_landmarks[34, :]-self.face_landmarks[264, :])
        head_heigth_proyection_px = np.linalg.norm(
            self.face_landmarks[10, :]-self.face_landmarks[152, :])

        #Projection correction
        head_width_px = head_width_proyection_px/np.cos(np.deg2rad(yaw))
        head_heigth_px = head_heigth_proyection_px/np.cos(np.deg2rad(pitch))
        head_distance_cm_w = (self.reference_head_width_cm *
                              self.focal_length_px/head_width_px) - self.nose_to_temple_correction
        head_distance_cm_h = (self.reference_head_heigth_cm *
                              self.focal_length_px/head_heigth_px) - self.nose_to_temple_correction
        head_distance_cm = (head_distance_cm_w+head_distance_cm_h)/2

        return head_distance_cm
    


    
    #### Calibration DataBase managment ####
    
    ### In the future, manage also initial head pose information
    
    def _saveCalibration(self):
        
        
        calibrationVariables = [self.calib_coeffs_leye, self.calib_coeffs_reye]
        
        pickle_out = open("DataBase/CalibrationData", "wb")
        pickle.dump(calibrationVariables, pickle_out)
        pickle_out.close()
        
    
    
    def _loadCalibration(self):
        
        pickle_in = open("DataBase/CalibrationData", "rb")
        calibrationVariables = pickle.load(pickle_in)
        pickle_in.close()
        
        self.calib_coeffs_leye = calibrationVariables[0]
        self.calib_coeffs_reye = calibrationVariables[1]

  


    #############################################
    #                                           #
    #              work in progress             #
    #                                           #
    #############################################
    
    def _estimateGazeAndProjection_v2(self, show=False):
        
        length = 120
        color = (0, 0, 255)
        thickness = 1
        if self.face_landmarks is not None and self.lPupil_center is not None and self.rPupil_center is not None:
            
            # Left eye
            eyeball_centre = np.mean(self.face_landmarks[[33, 133],:], axis=0) - [5, 0]
            
            eye_pos = self.lPupil_center*3+self.leye_corner
            i_x0, i_y0 = self.lPupil_center*3+self.leye_corner
            e_x0, e_y0 = eyeball_centre
            eyeball_radius = np.linalg.norm(self.face_landmarks[33,:]-self.face_landmarks[133,:])/2
            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                    -1.0, 1.0))
            
            cv2.circle(self.frame, tuple(map(int, eyeball_centre)), 3, (255, 0, 0), thickness=-1)
            pitchyaw = np.array([theta, phi])
            dx = -length * np.sin(pitchyaw[1])
            dy = -length * np.sin(pitchyaw[0])
            cv2.arrowedLine(self.frame, tuple(np.round(eye_pos).astype(np.int32)),
                           tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                           thickness, cv2.LINE_AA, tipLength=0.2)
            
            
            # Left eye
            eyeball_centre = np.mean(self.face_landmarks[[362, 263],:], axis=0)
            
            eye_pos = self.rPupil_center*3+self.reye_corner
            i_x0, i_y0 = self.rPupil_center*3+self.reye_corner
            e_x0, e_y0 = eyeball_centre
            eyeball_radius = np.linalg.norm(self.face_landmarks[362,:]-self.face_landmarks[263,:])/2
            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                    -1.0, 1.0))
            
            cv2.circle(self.frame, tuple(map(int, eyeball_centre)), 3, (255, 0, 0), thickness=-1)
            pitchyaw = np.array([theta, phi])
            dx = -length * np.sin(pitchyaw[1])
            dy = -length * np.sin(pitchyaw[0])
            cv2.arrowedLine(self.frame, tuple(np.round(eye_pos).astype(np.int32)),
                           tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                           thickness, cv2.LINE_AA, tipLength=0.2)

    def _estimateGazeAndProjection(self, show=False):
        ## Estimates for one eye
        
        if self.face_landmarks is not None and self.lPupil_center is not None and self.rPupil_center is not None:
            
            # left eye
            line_left = [self.face_landmarks[33,0], self.face_landmarks[33,1], self.face_landmarks[133,0], self.face_landmarks[133,1]]
            crossover_left = self._point_line(self.lPupil_center*3+self.leye_corner, line_left)
            yaw_mean = np.linalg.norm(self.face_landmarks[33,:]-self.face_landmarks[133,:])/2
            yaw_right = np.linalg.norm(self.face_landmarks[33,:]-crossover_left)
            yaw = (yaw_mean - yaw_right) / yaw_mean
            yaw_left = int(yaw * 71.58 + 14)
            
            pitch_dis = -np.linalg.norm((self.lPupil_center*3+self.leye_corner)-crossover_left)
            if (self.lPupil_center*3+self.leye_corner)[1] < crossover_left[1]:
                pitch_dis = -pitch_dis
            pitch_left = int(5 * pitch_dis)
            
            # right eye
            line_right = [self.face_landmarks[362,0], self.face_landmarks[362,1], self.face_landmarks[263,0], self.face_landmarks[263,1]]
            crossover_right = self._point_line(self.rPupil_center*3+self.reye_corner, line_right)
            yaw_mean = np.linalg.norm(self.face_landmarks[362,:]-self.face_landmarks[263,:])/2
            yaw_right = np.linalg.norm(self.face_landmarks[362,:]-crossover_left)
            yaw = (yaw_mean - yaw_right) / yaw_mean
            yaw_right = int(yaw * 71.58 + 0.7037)
            
            pitch_dis = -np.linalg.norm((self.rPupil_center*3+self.reye_corner)-crossover_right)
            if (self.rPupil_center*3+self.reye_corner)[1] < crossover_right[1]:
                pitch_dis = -pitch_dis
            pitch_right = int(1.497 * pitch_dis - 18.97)
            
            #Showing Arrow
            if show:
                pitchyaw = np.array([pitch_left, yaw_left])*np.pi/180
                nosetip = self.lPupil_center*3+self.leye_corner
                nosetip[0] = int(nosetip[0])
                nosetip[1] = int(nosetip[1])
                
                length = 120.0
                thickness = 1
                color = (0, 0, 255)
                dx = -length * np.sin(pitchyaw[1])
                dy = -length * np.sin(pitchyaw[0])
                cv2.arrowedLine(self.frame, tuple(np.round(nosetip).astype(np.int32)),
                               tuple(np.round([nosetip[0] + dx, nosetip[1] + dy]).astype(int)), color,
                               thickness, cv2.LINE_AA, tipLength=0.2)
                
                pitchyaw = np.array([pitch_right, yaw_right])*np.pi/180
                nosetip = self.rPupil_center*3+self.reye_corner
                nosetip[0] = int(nosetip[0])
                nosetip[1] = int(nosetip[1])
                
                length = 120.0
                thickness = 1
                color = (0, 0, 255)
                dx = -length * np.sin(pitchyaw[1])
                dy = -length * np.sin(pitchyaw[0])
                cv2.arrowedLine(self.frame, tuple(np.round(nosetip).astype(np.int32)),
                               tuple(np.round([nosetip[0] + dx, nosetip[1] + dy]).astype(int)), color,
                               thickness, cv2.LINE_AA, tipLength=0.2)

    
    def getEyesRotations(self):
        
        center_sphere_reye = (self.face_landmarks[362,:] + self.face_landmarks[263,:]) / 2
        radius_reye = np.linalg.norm(
            self.face_landmarks[362, :] - self.face_landmarks[263, :]) / 2
        radius_reye = radius_reye / np.cos(np.deg2rad(self.yaw))
        
        center_sphere_leye = (self.face_landmarks[33,:] + self.face_landmarks[133,:]) / 2
        radius_leye = np.linalg.norm(
            self.face_landmarks[33, :] - self.face_landmarks[133, :]) / 2
        # radius_leye = radius_leye / np.cos(np.deg2rad(self.yaw))
        
        radius_beye = (radius_leye + radius_reye) / 2
        
        #cv2.circle(self.frame, (int(center_sphere_reye[0]), int(center_sphere_reye[1])), int(radius_beye), (255,255,0), thickness=1)
        cv2.circle(self.frame, (int(center_sphere_leye[0]), int(center_sphere_leye[1])), int(radius_beye), (255,255,0), thickness=1)
        #cv2.circle(self.frame, (int(center_sphere_reye[0]), int(center_sphere_reye[1])), 1, (255,255,0), thickness=1)
        cv2.circle(self.frame, (int(center_sphere_leye[0]), int(center_sphere_leye[1])), 1, (255,255,0), thickness=1)
        
        cv2.circle(self.frame, (int(self.leye_corner[0]), int(self.leye_corner[1])), 1, (255,255,0), thickness=1)
        cv2.circle(self.frame, (int(self.reye_corner[0]), int(self.reye_corner[1])), 1, (255,255,0), thickness=1)
        
        
        if self.lPupil_center is not None:
            yaw_leye = np.rad2deg(np.arctan((center_sphere_leye[0] - (self.lPupil_center[0] + self.leye_corner[0]) ) / radius_leye))
            #print(center_sphere_leye[0], self.lPupil_center + self.leye_corner, radius_leye)
            #print(yaw_leye)
            yaw_leye = yaw_leye - 8  #Edu tenia que sumarle 8...
            
            i_x0, i_y0 = self.lPupil_center + self.leye_corner
            e_x0, e_y0 = center_sphere_leye
            theta = -np.arcsin(np.clip((i_y0 - e_y0) / radius_leye, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (radius_leye * -np.cos(theta)),
                                    -1.0, 1.0)) 
            current_gaze = np.array([0, np.deg2rad(yaw_leye)])
            #print(current_gaze)
            
            pitchyaw = current_gaze
            nosetip = self.lPupil_center+self.leye_corner
            nosetip[0] = int(nosetip[0])
            nosetip[1] = int(nosetip[1])
            
            length = 120.0
            thickness = 1
            color = (0, 0, 255)
            dx = -length * np.sin(pitchyaw[1])
            dy = -length * np.sin(pitchyaw[0])
            cv2.arrowedLine(self.frame, tuple(np.round(nosetip).astype(np.int32)),
                           tuple(np.round([nosetip[0] + dx, nosetip[1] + dy]).astype(int)), color,
                           thickness, cv2.LINE_AA, tipLength=0.2)
            
    