#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:57:34 2021

@author: josesolla
"""


import cv2
import numpy as np

import collections

import mediapipe as mp
import time
import onnxruntime



DRAW_LMS = False
highlighted_indices = []
metaShow = True

## Eye indices. 133 is left inner eye
leye_lms = np.array([  7,  33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161,
       163, 173, 246])
## 362 is right inner eye
reye_lms = np.array([249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388,
       390, 398, 466])

## Landmark smoothing
smoothing_window_size = 10
smoothing_coefficient_decay = 0.5
smoothing_coefficients = None


class Detector(object):
    
    
    def __init__(self, gazeTracker, detect_conf=0.5, track_conf=0.5):
        
        self.gazeTracker = gazeTracker

        # counter of face tracking consecutive losses
        self.face_ok_counter = 0 

        # Debbuging purposes
        self.cont = 0
        
        # Mediapipe forming objects
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=detect_conf, min_tracking_confidence=track_conf)
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        
        ## History of landmarks for smoothing
        self.landmark_entries_window = collections.deque(maxlen=smoothing_window_size)
        
        onnx_model = './models/elg_onnx_tf2onnx_bilinear.onnx'
        self.sess = onnxruntime.InferenceSession(onnx_model)
        self.sess_input_name = self.sess.get_inputs()[0].name
        
        # ####### ONNX MediaPipe TFLite iris landmarks detector
        # onnx_model = 'models/iris_landmark.onnx'
        # self.iris_sess = onnxruntime.InferenceSession(onnx_model)
        
        
        ### Timers
        # self.accum_mediapipe = 0
        # self.accum_postmediapipe = 0
        # self.accum_onnx = 0
        # self.accum_onnxdrawing = 0
        
    
    def detect(self, show=False):
        
        """ MediaPipe Face detection and landmark prediction
        
        """
        
        ## MediaPipe pre-process: Convert BGR 2 RGB
        image = cv2.cvtColor(self.gazeTracker.frame, cv2.COLOR_BGR2RGB)
        
        # Infere
        # start = time.time()
        results = self.face_mesh.process(image)
        # self.accum_mediapipe = self.accum_mediapipe + (time.time()-start)
        
        # If correctly infered
        if results.multi_face_landmarks:
            
            lms = []
            
            # start = time.time()
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                lms.append((landmark.x, landmark.y))

            lms = np.asarray(lms)
           
            
            
            # Convert to pixel landmarks
            shape = np.array([image.shape[1], image.shape[0]])
            self.gazeTracker.face_landmarks = self._normalized_to_pixel_coordinates(lms, 
                                                                                    shape)
            
            
            
            # Append to landmarks history and smooth
            # self.landmark_entries_window.append(self.gazeTracker.face_landmarks)
            # smooth_landmarks(smoothing_coefficients, self.landmark_entries_window)
            # self.gazeTracker.face_landmarks = self.landmark_entries_window[-1]
            
            # Draw 
            if show and DRAW_LMS: 
                #self._draw_landmarks()
                self.mp_drawing.draw_landmarks(
                    image=self.gazeTracker.frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=self.mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
             
            # Highlight wanted landmarks
            self._drawLandmarks(highlighted_indices)
            
            ## Get eye frames 
            left_eye, right_eye = self._extractEyes(self.gazeTracker.frame, 
                                                    self.gazeTracker.face_landmarks[:,0:2])
            
            # self.accum_postmediapipe = self.accum_postmediapipe + (time.time()-start)
            # Set gazeTracker eye frames
            self.gazeTracker.left_eye = left_eye
            self.gazeTracker.right_eye = right_eye
            
            # GazeML Keras inference
            # start = time.time()
            result_left, result_right, lPupil, rPupil = self._detectGazeMLKeras(left_eye, right_eye, confidence=0.8, show = show)
            
            # self.accum_onnx = self.accum_onnx + (time.time()-start)
    
            self.gazeTracker.lPupil_center = lPupil
            self.gazeTracker.rPupil_center = rPupil 
            
            if show and metaShow:
                
                #if self.gazeTracker.calib_coeffs_leye is None:
                
                # start = time.time()
                leftt = cv2.resize(result_left, (240, 120))
                rightt = cv2.resize(result_right, (240, 120))
                
                # leftt = cv2.cvtColor(leftt, cv2.COLOR_BGR2GRAY)
                # rightt = cv2.cvtColor(rightt, cv2.COLOR_BGR2GRAY)
                
                # leftt = cv2.equalizeHist(leftt)
                # rightt = cv2.equalizeHist(rightt)
                
                # leftt = cv2.cvtColor(leftt, cv2.COLOR_GRAY2BGR)
                # rightt = cv2.cvtColor(rightt, cv2.COLOR_GRAY2BGR)
                
                
                self.gazeTracker.frame[:120, :240, :] = leftt
                self.gazeTracker.frame[:120, 240:480, :] = rightt
                # self.accum_onnxdrawing = self.accum_onnxdrawing + (time.time()-start)
                

    def _normalized_to_pixel_coordinates(self, lms, shape):
        
        
        ## Check which normalized values are valid (between 0 and 1)
        lms_copy = lms.copy()
        
        
        cond1 = np.logical_and(lms_copy[:,0]>=0, lms_copy[:,0]<=1)
        cond2 = np.logical_and(lms_copy[:,1]>=0, lms_copy[:,1]<=1)
        
        idx_keep = np.where(np.logical_and(cond1, cond2))[0]
        
        # Multiply per image shape
        lms_copy[idx_keep,0:2] = np.floor(lms_copy[idx_keep,0:2]*(shape-1)).astype(int)
        
        return lms_copy

    def _extractEyes(self, img, lms, xMargin=0.2, yMargin=0.75): #margin to 70 if using medipipe
        
        # Calculate pixel margin corresponding to eye frame size
        #percentageMargin = 5
        
    
        # Mediapipe eyes landamrks

        left_eye_coords  = lms[leye_lms,:]
        right_eye_coords = lms[reye_lms,:]
        
        left_eye_upleft       = np.min(left_eye_coords, axis=0)
        
        left_eye_bottomright  = np.max(left_eye_coords, axis=0)
        leye_x_margin = int((left_eye_bottomright[0]-left_eye_upleft[0])*xMargin)
        leye_y_margin = int((left_eye_bottomright[1]-left_eye_upleft[1])*yMargin)
        extraMargin = np.array([leye_x_margin, leye_y_margin])
        self.gazeTracker.leye_corner = left_eye_upleft-extraMargin
        
        right_eye_upleft       = np.min(right_eye_coords, axis=0)
        right_eye_bottomright  = np.max(right_eye_coords, axis=0)
        reye_x_margin = int((right_eye_bottomright[0]-right_eye_upleft[0])*xMargin)
        reye_y_margin = int((right_eye_bottomright[1]-right_eye_upleft[1])*yMargin)
        extraMargin = np.array([reye_x_margin, reye_y_margin])
        self.gazeTracker.reye_corner = right_eye_upleft-extraMargin

        

        left_eye  = img[int(left_eye_upleft[1]-extraMargin[1]):int(left_eye_bottomright[1]+extraMargin[1]), int(left_eye_upleft[0]-extraMargin[0]):int(left_eye_bottomright[0]+extraMargin[0])]
        right_eye = img[int(right_eye_upleft[1]-extraMargin[1]):int(right_eye_bottomright[1]+extraMargin[1]), int(right_eye_upleft[0]-extraMargin[0]):int(right_eye_bottomright[0]+extraMargin[0])]
    
        return left_eye, right_eye

    def _extractEyes_v2(self, img, lms, extraMargin=15): #margin to 70 if using medipipe
        
        # Calculate pixel margin corresponding to eye frame size
        #percentageMargin = 5
        
    
        # Mediapipe eyes landamrks

        left_eye_coords  = lms[leye_lms,:]
        right_eye_coords = lms[reye_lms,:]
        
        left_eye_upleft       = np.min(left_eye_coords, axis=0)
        self.gazeTracker.leye_corner = left_eye_upleft-extraMargin
        left_eye_bottomright  = np.max(left_eye_coords, axis=0)
        
        right_eye_upleft       = np.min(right_eye_coords, axis=0)
        self.gazeTracker.reye_corner = right_eye_upleft-extraMargin
        right_eye_bottomright  = np.max(right_eye_coords, axis=0)
        
        left_eye  = img[int(left_eye_upleft[1]-extraMargin):int(left_eye_bottomright[1]+extraMargin), int(left_eye_upleft[0]-extraMargin):int(left_eye_bottomright[0]+extraMargin)]
        right_eye = img[int(right_eye_upleft[1]-extraMargin):int(right_eye_bottomright[1]+extraMargin), int(right_eye_upleft[0]-extraMargin):int(right_eye_bottomright[0]+extraMargin)]
    
        return left_eye, right_eye
    
    
    def _extractEyes_v3(self, img, lms):
        
        
        oh, ow = (36, 60)
        #oh, ow = (36, 60) Intended for webcam in original project
        
        # [(36, 39, True), (42, 45, False)] Old for dlib, check lms in mediapipe
        # [(33, 133, True), (263, 362, False)]
        
        for corner1, corner2, is_left in [(33, 133, True), (263, 362, False)]:
            x1, y1 = lms[corner1, :]
            x2, y2 = lms[corner2, :]
            eye_width = 1.5 * np.linalg.norm(lms[corner1, :] - lms[corner2, :])
            if eye_width == 0.0:
                continue
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            
            # Centre image on middle of eye
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # Rotate to be upright
            roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
            rotate_mat = np.asmatrix(np.eye(3))
            cos = np.cos(-roll)
            sin = np.sin(-roll)
            rotate_mat[0, 0] = cos
            rotate_mat[0, 1] = -sin
            rotate_mat[1, 0] = sin
            rotate_mat[1, 1] = cos
            inv_rotate_mat = rotate_mat.T

            # Scale
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            # Centre image
            centre_mat = np.asmatrix(np.eye(3))
            centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            inv_centre_mat = np.asmatrix(np.eye(3))
            inv_centre_mat[:2, 2] = -centre_mat[:2, 2]
            

            # Get rotated and scaled, and segmented image
            transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
            
            inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                 inv_centre_mat)
            
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eye_image = cv2.warpAffine(gray_frame, transform_mat[:2, :], (ow, oh))

            if is_left:
                left_eye = eye_image.copy().astype(np.uint8)
                self.gazeTracker.leye_corner =  np.array([cx-(eye_image.shape[1]/2), cy-(eye_image.shape[0]/2)])
                
                cv2.circle(self.gazeTracker.frame, tuple(map(int, self.gazeTracker.leye_corner)), 6, (255, 0, 0), thickness=-1)
                
            else:
                right_eye = eye_image.copy().astype(np.uint8)
                self.gazeTracker.reye_corner =  np.array([cx-(eye_image.shape[1]/2), cy-(eye_image.shape[0]/2)])
                cv2.circle(self.gazeTracker.frame, tuple(map(int, self.gazeTracker.reye_corner)), 6, (255, 0, 0), thickness=-1)
                #eye_image = np.fliplr(eye_image)
                
            
        return left_eye, right_eye
         

    
    def _detectIrisMediaPipe(self, left_eye, right_eye, show=True):
        
        
        input_left = cv2.resize(left_eye, (64, 64))[np.newaxis, ...]
        input_left = (np.float32(input_left) - 0.0) / 255.0
        
        input_right = cv2.resize(right_eye, (64, 64))[np.newaxis, ...]
        input_right = (np.float32(input_right) - 0.0) / 255.0
        
        input_name = self.iris_sess.get_inputs()[0].name
        
        # Run inference
        left_onnx_eye, left_onnx_iris = self.iris_sess.run(None, {input_name: input_left})
        right_onnx_eye, right_onnx_iris = self.iris_sess.run(None, {input_name: input_right})
        
        
        scale_x_l = left_eye.shape[1]/64
        scale_y_l = left_eye.shape[0]/64
        
        left_onnx_iris[0][::3] = left_onnx_iris[0][::3]*scale_x_l
        left_onnx_iris[0][1::3] = left_onnx_iris[0][1::3]*scale_y_l
        
        scale_x_r = right_eye.shape[1]/64
        scale_y_r = right_eye.shape[0]/64
        
        right_onnx_iris[0][::3] = right_onnx_iris[0][::3]*scale_x_r
        right_onnx_iris[0][1::3] = right_onnx_iris[0][1::3]*scale_y_r
        
        
        
        result_left = left_eye.copy().astype(np.uint8)
        if show:
            radius = int(np.abs(np.linalg.norm(left_onnx_iris[0][::3][0]-left_onnx_iris[0][::3][1])))
            cv2.circle(result_left, (int(left_onnx_iris[0][::3][0]), int(left_onnx_iris[0][1::3][0])), radius, (255, 0, 0), thickness=4)
            for i in range(5):
                if i==0:
                    cv2.circle(result_left, (int(left_onnx_iris[0][::3][i]), int(left_onnx_iris[0][1::3][i])), 3, (0, 255, 0), thickness=-1)
                else:   
                    cv2.circle(result_left, (int(left_onnx_iris[0][::3][i]), int(left_onnx_iris[0][1::3][i])), 3, (0, 0, 255), thickness=-1)

        result_right = right_eye.copy().astype(np.uint8)
        if show:
            radius = int(np.abs(np.linalg.norm(right_onnx_iris[0][::3][0]-right_onnx_iris[0][::3][1])))
            cv2.circle(result_right, (int(right_onnx_iris[0][::3][0]), int(right_onnx_iris[0][1::3][0])), radius, (255, 0, 0), thickness=4)
            for i in range(5):
                if i==0:
                    cv2.circle(result_right, (int(right_onnx_iris[0][::3][i]), int(right_onnx_iris[0][1::3][i])), 3, (0, 255, 0), thickness=-1)
                else:   
                    cv2.circle(result_right, (int(right_onnx_iris[0][::3][i]), int(right_onnx_iris[0][1::3][i])), 3, (0, 0, 255), thickness=-1)


        return result_left, result_right, None, None

    def _detectGazeMLKeras(self, left_eye, right_eye, confidence=0.8, show=True):
        
        result_left = None
        result_right = None
        
        
        

        # Preprocess for NN
        inp_left = auto_white_balance(left_eye)
        inp_left = cv2.cvtColor(inp_left, cv2.COLOR_BGR2GRAY)
        # inp_left = left_eye.copy().astype(np.uint8) #cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        # left_eye = cv2.cvtColor(left_eye, cv2.COLOR_GRAY2BGR)
        inp_left = cv2.equalizeHist(inp_left)
        #leye_processed = cv2.cvtColor(inp_left, cv2.COLOR_GRAY2BGR)
        inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]
        
        inp_right = auto_white_balance(right_eye)
        inp_right = cv2.cvtColor(inp_right, cv2.COLOR_BGR2GRAY)
        # inp_right = right_eye.copy().astype(np.uint8) #cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        # right_eye = cv2.cvtColor(right_eye, cv2.COLOR_GRAY2BGR)
        inp_right = cv2.equalizeHist(inp_right)
        inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]
        
        ## NN output
        input_array = np.concatenate([inp_left, inp_right], axis=0).astype(np.float32)
        
        
        
        #s = time.time()
        pred_left, pred_right = self.sess.run(None, {self.sess_input_name: input_array/255 * 2 - 1})[0]
        #print(time.time()-s)
        
        
        # Get landmarks
        lms_left = calculate_landmarks(pred_left)
        lms_right = calculate_landmarks(pred_right)

        ## Filter usable landmarks. ONLY interested in iris landmarks!
        heatmaps_amax_left = np.amax(pred_left.reshape(-1,18), axis=0)
        heatmaps_amax_right = np.amax(pred_right.reshape(-1,18), axis=0)
        
        if np.all(heatmaps_amax_left[8:16] > confidence) and np.all(heatmaps_amax_right[8:16] > confidence):
            
            if show: 
                #if self.gazeTracker.calib_coeffs_leye is None:
                result_left = draw_pupil(left_eye, inp_left, lms_left)
            if show: 
                #if self.gazeTracker.calib_coeffs_reye is None:
                result_right = draw_pupil(right_eye, inp_right, lms_right)
        
            ## Pupil landmark!
            #scale = 3
            shape = left_eye.shape[0:2]
            scale = np.array([180, 108])/np.array([shape[1], shape[0]])
            left_pupil = lms_left[0,16,:]/scale
            
            shape = right_eye.shape[0:2]
            scale = np.array([180, 108])/np.array([shape[1], shape[0]])
            right_pupil = lms_right[0,16,:]/scale
        
            return result_left, result_right, left_pupil*3, right_pupil*3
    
        else:
            result_left = left_eye
            result_right = right_eye
            return result_left, result_right, None, None
    
    
    def _draw_landmarks(self):    

        for (x,y) in self.gazeTracker.face_landmarks:
            cv2.circle(self.gazeTracker.frame, (int(x),int(y)), 4, (0, 255, 0), thickness=-1)
    
    def _drawLandmarks(self, indices_print, color=(0, 0, 255)):
        
        """
            indices_print: lista de indices para imprimir o resaltar
        """
        
        
        for index, lm in enumerate(self.gazeTracker.face_landmarks):
            
            if index in indices_print:
                (x,y) = lm[0:2]
                
                cv2.circle(self.gazeTracker.frame, (int(x), int(y)), 1, color, thickness=-1)
            
    
    
    def _faceMeshTFLite(self, img):
        
        
        
        
        pass
        
        
    
def smooth_landmarks(smoothing_coefficients, landmark_entries_window):
    """If there are previous landmark detections, try to smooth current prediction."""
    # Cache coefficients based on defined sliding window size
    if smoothing_coefficients is None:
        coefficients = np.power(smoothing_coefficient_decay,
                                list(reversed(list(range(smoothing_window_size)))))
        coefficients /= np.sum(coefficients)
        smoothing_coefficients = coefficients.reshape(-1, 1)

    # Get a window of frames
    #current_index = _indices.index(frame['frame_index'])
    a = len(landmark_entries_window)
    if a < smoothing_window_size:
        """If not enough frames for smoothing window."""
        return
    # window_indices = _indices[a:current_index + 1]
    # window_frames = [frames[idx] for idx in window_indices]
    # window_num_landmark_entries = np.array([len(f['landmarks']) for f in window_frames])
    if np.any(landmark_entries_window == None):
        """Any frame has zero faces detected."""
        return 

    # Apply coefficients to landmarks in window
    window_landmarks = np.asarray(landmark_entries_window)
    smoothed_landmarks = np.sum(
        np.multiply(window_landmarks.reshape(smoothing_window_size, -1),
                    smoothing_coefficients),
        axis=0,
    ).reshape(len(landmark_entries_window[-1]), -1, 2)
    
    landmark_entries_window[-1] = smoothed_landmarks[:,0].astype(int)
    

    
def calculate_landmarks(x, beta=5e1):
    def np_softmax(x, axis=1):
        t = np.exp(x)
        a = np.exp(x) / np.sum(t, axis=axis).reshape(-1,1)
        return a

    if len(x.shape) < 4:
        x = x[None, ...]
    h, w = x.shape[1:3]
    ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                 np.linspace(0, 1.0, num=h, endpoint=True),
                                 indexing='xy')
    ref_xs = np.reshape(ref_xs, [-1, h*w])
    ref_ys = np.reshape(ref_ys, [-1, h*w])

    # Assuming N x 18 x 45 x 75 (NCHW)
    beta = beta
    x = np.transpose(x, (0, 3, 1, 2))
    x = np.reshape(x, [-1, 18, h*w])
    x = np_softmax(beta * x, axis=-1)
    lmrk_xs = np.sum(ref_xs * x, axis=2)
    lmrk_ys = np.sum(ref_ys * x, axis=2)

    # Return to actual coordinates ranges
    return np.stack([lmrk_xs * (w - 1.0) + 0.5, lmrk_ys * (h - 1.0) + 0.5], axis=2)  # N x 18 x 2


def draw_pupil(im, inp_im, lms):
    draw = im
    draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
    pupil_center = np.zeros((2,))
    pnts_outerline = []
    pnts_innerline = []
    stroke = inp_im.shape[1] // 12 + 1
    for i, lm in enumerate(np.squeeze(lms)):
        #print(lm)
        y, x = int(lm[0]*3), int(lm[1]*3)

        if i < 8:
            continue
            draw = cv2.circle(draw, (y, x), 2, (255, 255, 0), -1)
            pnts_outerline.append([y, x])
        elif i < 16:
            draw = cv2.circle(draw, (y, x), 2, (0, 255, 255), -1)
            pnts_innerline.append([y, x])
            pupil_center += (y,x)
        elif i < 17:
            draw = cv2.drawMarker(draw, (y, x), (255,200,200), markerType=cv2.MARKER_CROSS, markerSize=2, thickness=1, line_type=cv2.LINE_AA)
        else:
            draw = cv2.drawMarker(draw, (y, x), (255,255,255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA)
    pupil_center = (pupil_center/8).astype(np.int32)
    draw = cv2.circle(draw, (pupil_center[0], pupil_center[1]), 2, (0, 255, 0), -1)        
    draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(255, 255, 0), thickness=2)
    draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(0, 255, 255), thickness=2)
    return draw


def auto_white_balance(image):
    """

    Parameters
    ----------
    image : ARRAY OF UINT8
        THIS FUNCTION IS OPTIONAL. IT CAN BE CALLED 
        BEFORE ANALYZING THE IMAGE.
        

    Returns
    -------
    result : ARRAY OF UINT8
        IT WILL RETURN THE ORIGINAL IMAGE WITH AN 
        AUTOMATIC WHITE BALANCE APPLIED TO IT.

    """
    
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

