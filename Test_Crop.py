from lib.Dataset import ConstructionDataset
from lib.utils import parse_yaml

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data
import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
    
    
    
    class_dict = {
        0: "worker",
        1: "suspended load",
        2: "static crane",
        3: "crane",
        4: "roller",
        5: "bulldozer",
        6: "excavator",
        7: "truck",
        8: "loader",
        9: "pump truck",
        10: "concrete mixer",
        11: "pile driving",
        12: "forklift",
    }

    print("Loading all detected objects in dataset...")

    # Initialise dataset stuff
    batch_size = 1 # Only for testing purposes
    safe_distancing = config['General']['safe_dist']

    consDataset = ConstructionDataset(config, crops=True)

    params = {'batch_size': batch_size,
              'shuffle':True,
              'num_workers': 6}

    generator = data.DataLoader(consDataset, **params)

    count=0
    
    
    
    print('Displaying')
    for local_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, new_vps_3d, new_vps_2d, indexed_orientation, indexed_dims in generator:
        # print(indexed_box2d)
        x_cent, y_cent, local_width, local_height = float(indexed_box2d[0][0]),float(indexed_box2d[1][0]), float(indexed_box2d[2][0]),float(indexed_box2d[3][0])
        print(local_width/960*224, local_height/960*224)
        label=(class_dict[int(indexed_clas[0])])
        
        plt.imshow(local_crop[0])
        plt.show()
        # Applying the Canny Edge filter 
        # img = np.array(local_crop)
        
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # edge = cv2.Canny(blurred, t_lower, t_upper, apertureSize=3) 
        
        # cv2.imshow(label,edge)
        # cv2.imshow(f'{label}1',blurred)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__=='__main__':
    config=parse_yaml('config.yaml')
    main(config)
