"""
Functions to read from files
TODO: move the functions that read label from Dataset into here
"""
import numpy as np


# File.py referenced from https://github.com/skhadem/3D-BoundingBox
def get_P(cab_f):
    for line in open(cab_f):
        if 'P_rect_02' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            return_matrix = np.zeros((3,4))
            return_matrix = cam_P.reshape((3,4))
            return return_matrix
    
    raise Exception('Unexpected error occured')

def file_not_found(filename):
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    exit()


