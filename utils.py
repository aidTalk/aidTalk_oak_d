#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:51:59 2021

@author: josesolla
"""


import numpy as np



FACE_CONNECTIONS = [
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133)
]

## MediaPipe eye landmarks

reye_lms = np.array([  7,  33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161,
       163, 173, 246])

leye_lms = np.array([249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388,
       390, 398, 466])




smoothing_window_size = 10
smoothing_coefficient_decay = 0.5
smoothing_coefficients = None

def calculate_smoothed_landmarks(frame, _indices, smoothing_coefficients, frames):
    """If there are previous landmark detections, try to smooth current prediction."""
    # Cache coefficients based on defined sliding window size
    if smoothing_coefficients is None:
        coefficients = np.power(smoothing_coefficient_decay,
                                list(reversed(list(range(smoothing_window_size)))))
        coefficients /= np.sum(coefficients)
        smoothing_coefficients = coefficients.reshape(-1, 1)

    # Get a window of frames
    current_index = _indices.index(frame['frame_index'])
    a = current_index - smoothing_window_size + 1
    if a < 0:
        """If slice extends before last known frame."""
        return
    window_indices = _indices[a:current_index + 1]
    window_frames = [frames[idx] for idx in window_indices]
    window_num_landmark_entries = np.array([len(f['landmarks']) for f in window_frames])
    if np.any(window_num_landmark_entries == 0):
        """Any frame has zero faces detected."""
        return
    if not np.all(window_num_landmark_entries == window_num_landmark_entries[0]):
        """Not the same number of faces detected in entire window."""
        return

    # Apply coefficients to landmarks in window
    window_landmarks = np.asarray([f['landmarks'] for f in window_frames])
    frame['smoothed_landmarks'] = np.sum(
        np.multiply(window_landmarks.reshape(smoothing_window_size, -1),
                    smoothing_coefficients),
        axis=0,
    ).reshape(window_num_landmark_entries[-1], -1, 2)
    

import collections

smoothing_window_size = 10
smoothing_coefficient_decay = 0.5
smoothing_coefficients = None

# Landmarks list (igual esto para dentro del detector)
landmark_entries_window = collections.deque(maxlen=smoothing_window_size)

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
        """If slice extends before last known frame."""
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
    
    
    

# Al hacer la prueba hacer
new_lms = 0
landmark_entries_window.append(new_lms)

smooth_landmarks(smoothing_coefficients, landmark_entries_window)


smoothed = smooth_landmarks(0, 0, 1, landmark_entries_window)
if smoothed:
    landmark_entries_window[-1] = smoothed



    
    