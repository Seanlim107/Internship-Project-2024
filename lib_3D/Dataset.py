import cv2
import numpy as np
import os
import random

import torch
from torchvision import transforms

from lib_3D.File import *

class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None, resize_size=960):

        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_P(proj_matrix)
            # proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.resize_size = tuple([resize_size, resize_size])
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class
        

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        # crop image
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        crop = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        crop = cv2.resize(src = crop, dsize=self.resize_size, interpolation=cv2.INTER_CUBIC)

        # temp_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        temp_img = cv2.resize(crop, self.resize_size)/255.0
        temp_img = np.transpose(temp_img, (2,0,1))
        temp_img = torch.tensor(temp_img, dtype=torch.float32)
        batch = temp_img

        return batch
