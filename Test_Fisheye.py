from lib.Dataset import ConstructionDataset


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data
import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2
import numpy as np
from lib.Camera import Camera
from lib.utils import parse_yaml

from lu_vp_detect import VPDetection

import yaml

import os

def main(config):

    # hyper parameters
    epochs = 100
    batch_size =1
    
    # Setting parameter values 
    t_lower = 50  # Lower Threshold 
    t_upper = 200  # Upper threshold 
    
    config_cam = config['Camera']


    print("Loading all detected objects in dataset...")

    dataset = ConstructionDataset(config, crops=True)

    params = {'batch_size': batch_size,
              'shuffle':True,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    count=0
    
    
    
    print('Displaying')
    for _, _, _, _, local_image, _ in generator:
        
        # Applying the Canny Edge filter 
        # crop = np.array(local_crop[0])
        img = np.array(local_image)
        cam = Camera(use_own = False, img=img, distortion_coef=config_cam['distortion_coef'], fx=config_cam['fx'], fy=config_cam['fy'], cx=config_cam['cx'], cy=config_cam['cy'])
        undistorted_img = cam.remove_fisheye(img)
        
        # Display the original and undistorted images
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 2, 2)
        plt.title('Undistorted Image')
        plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))

        plt.show()

if __name__=='__main__':
    config=parse_yaml('config.yaml')
    main(config)
