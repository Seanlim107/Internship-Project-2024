from lib.utils import parse_yaml
from lib.Camera import Camera

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

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

class VPD:
    def __init__(self, config):
        # Drawing parameters
        self.Color_palette = 255 * np.eye(3)
        self.Color_palette = self.Color_palette[:, ::-1].astype(int).tolist()
        self.config_cam = config['Camera']
        config_vp = config['VP_Detector']
        self.remove_fisheye = self.config_cam['remove_fisheye']
        
        self.length_thresh = config_vp['length_thresh']
        self.principal_point = config_vp['principal_point']
        self.focal_length = config_vp['focal_length']
        self.seed = config['General']['seed']
        
        self.vpd = VPDetection(self.length_thresh, focal_length=self.focal_length, seed=self.seed)
        
    def get_vanishing_points(self, img):
        vps = self.vpd.find_vps(img)
        return vps
    
    def get_vanishing_points_2d(self, img):
        vps = self.vpd.find_vps(img)
        vps_2d = self.vpd.vps_2D[:3]
        return vps_2d
    
    def draw_line(self, image, point1, point2, color=(0, 0, 0), thickness=1):
        point1 = tuple(map(int, point1))
        point2 = tuple(map(int, point2))
        cv2.line(image, point1, point2, color, thickness)
    
    # Grid Drawing for visualization (Referenced with ChatGPT)
    def draw_grid(self, image, vp1, vp2, vp3, num_lines=20):
        height, width = image.shape[0:2]
        print(height, width)
        # print(f'height grid {height}, width grid {width}')
        color_vp1, color_vp2, color_vp3 = self.Color_palette
        
        # vp1=vp1*np.array([width,height])
        # vp2=vp2*np.array([width,height])
        # vp3=vp3*np.array([width,height])
        
        # print(vp1, vp2, vp3)

        # Create evenly spaced points along the edges of the image
        def create_edge_points(start, end, num_points):
            return [start + (end - start) * i / num_points for i in range(num_points + 1)]

        # Horizontal edges (top and bottom)
        top_edge = create_edge_points(np.array([0, 0]), np.array([width, 0]), num_lines)
        bottom_edge = create_edge_points(np.array([0, height]), np.array([width, height]), num_lines)

        # Vertical edges (left and right)
        left_edge = create_edge_points(np.array([0, 0]), np.array([0, height]), num_lines)
        right_edge = create_edge_points(np.array([width, 0]), np.array([width, height]), num_lines)

        # Draw lines from edges to vp1
        for point in top_edge + bottom_edge + left_edge + right_edge:
            self.draw_line(image, point, vp1, color=tuple(color_vp1))

        # Draw lines from edges to vp2
        for point in top_edge + bottom_edge + left_edge + right_edge:
            self.draw_line(image, point, vp2, color=tuple(color_vp2))

        # Draw lines from edges to vp3
        for point in top_edge + bottom_edge + left_edge + right_edge:
            self.draw_line(image, point, vp3, color=tuple(color_vp3))

    def get_vp_loc(self, image, vps_2d, vps_3d):
        # Function to define the axes of the vanishing points
        height, width = image.shape[1:3]
        vps_2d_calc = vps_2d
        # vps_2d_calc = vps_2d*np.array([width,height])
        mid_lower_point = [width / 2, height]

        distance = np.linalg.norm(vps_2d_calc - mid_lower_point, axis=1)
        threshold = height // 2  # Any VP located above this threshold vertically is set to infinity
        filtered_distance = np.where(vps_2d_calc[:, 1] < threshold, np.inf, distance)

        vp_z = np.argmin(filtered_distance)

        # Get y component
        temp_arr = np.delete(vps_2d_calc, vp_z, axis=0)
        dists = np.column_stack([temp_arr[:,0],np.array([width//2,height//2])])
        dists = np.linalg.norm(dists, axis=1).reshape(-1, 1)
        vp_y = np.argmin(dists)
        # vp_y = np.argmin(np.abs(temp_arr[:,0]))
        dists = temp_arr - np.array(image.shape[:-1][::-1]) // 2
        dists /= np.linalg.norm(dists, axis=1).reshape(-1, 1)
        vp_y = np.argmax(np.dot(dists, np.array([0, -1])))
        if vp_y >= vp_z:
            vp_y += 1

        # Get x component
        vp_x = [i for i in range(3) if (i != vp_z) and (i != vp_y)][0]
        vps_3d = vps_3d[[vp_x, vp_y, vp_z]]
        vps_2d = vps_2d[[vp_x, vp_y, vp_z]]

        return vps_2d, vps_3d

def main(config):
    # Hyperparameters
    batch_size = 1
    
    print("Loading all detected objects in dataset...")
    #Dataset
    config_yoloprep = config['Dataset']['Construction']
    mainpath = config_yoloprep['main_dir']
    imgdirname = config_yoloprep['img_dir_name']
    labeldirname = config_yoloprep['lab_dir_name']
    batchdirname = config_yoloprep['Batch_dir_name']
    start_num_batch = config_yoloprep['start_batch_num']
    end_num_batch = config_yoloprep['end_batch_num']
    moving_dir = config_yoloprep['moving_dir']
    img_ext = config_yoloprep['img_ext']
    label_ext = config_yoloprep['label_ext']
    train_size = config_yoloprep['train_size']
    test_size = config_yoloprep['test_size']
    valid_size = config_yoloprep['valid_size']
    init = config_yoloprep['init']
    resize_size = tuple([config['Dataset']['Construction']['resize_size'], config['Dataset']['Construction']['resize_size']])
    seed = config['General']['seed']
    modes = np.argmax(np.array([config['General']['use_scale'], config['General']['use_cam']]))
    
    
    consDataset = ConstructionDataset(config, crops=True)

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}
    generator = data.DataLoader(consDataset, **params)

    my_vpd = VPD(config)
    
    print('Displaying')
    count = 0
    for local_crop, local_crop_tensor, local_box, local_class, local_img, local_img_tensor in generator:
        curr_img = np.array(local_img[0])
        print(curr_img.shape)
        # print(curr_img)
        
        if my_vpd.remove_fisheye:
            cam = Camera(use_own=False, img=curr_img, distortion_coef=config['Camera']['distortion_coef'],
                         fx=config['Camera']['fx'], fy=config['Camera']['fy'], cx=config['Camera']['cx'], cy=config['Camera']['cy'])
            curr_img = cam.remove_fisheye(curr_img)
        else:
            cam = None

        vps_3d = my_vpd.get_vanishing_points(curr_img)
        vps_2d = my_vpd.get_vanishing_points_2d(curr_img)
        print(vps_3d)
        print(vps_2d)
        vps_2d, vps_3d = my_vpd.get_vp_loc(curr_img, vps_2d, vps_3d)
        vp1, vp2, vp3 = vps_2d
        print(vp1, vp2, vp3)
        my_vpd.draw_grid(curr_img, vp1, vp2, vp3)
        
        plt.imshow(curr_img)
        plt.show()
        
        count += 1

if __name__ == '__main__':
    config = parse_yaml('config.yaml')
    main(config)
