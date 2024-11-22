# Load a pre-trained YOLOv10n model
# from ultralytics import YOLOv10
from ultralytics import YOLO
from lib.Dataset import ConstructionDataset
from lib.Camera import Camera
from lib.utils import parse_yaml, estimate_distance_2d, draw_lines, estimate_distance_centers_3d
import torch
import os
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib.VP_Detector import VPD

def main(config):
    #Initialization
    filedir = os.path.dirname(__file__)
    # weightpath = os.path.join(filedir, 'runs/detect/train12/weights/best.pt')
    # model = YOLOv10(weightpath)
    # weightpath = os.path.join(filedir, 'runs/detect/train/weights/best.pt')
    # model = YOLO(weightpath)
    # lookup = config['Classes']
    config_cam = config['Camera']
    
    batch_size = 1 # Only for testing purposes
    # safe_distancing = config['General']['safe_dist']
    
    
    consDataset = ConstructionDataset(config, crops=False)
    

    params = {'batch_size': batch_size,
                'shuffle':True,
                'num_workers': 6}
    torch.manual_seed(config['General']['seed'])
    generator = data.DataLoader(consDataset, **params)
    count = 0
    my_vpd = VPD(config)
    
    # help = os.path.join('Construction Dataset/Images/Batch3/CES - Crane Tower_ch12_20231111074627_20231111074703_0001.jpg')
    # picture = consDataset.getimage(help)
    # picture = consDataset.format_img(help2)
    # temp_img_ori = np.array(picture)
    # plt.imshow(temp_img_ori)
    # plt.show()
    # vps_3d = my_vpd.get_vanishing_points(temp_img_ori)
    # print(vps_3d)
    
    
    
    # for (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, local_img, indexed_img_tensor, new_vps_3d, new_vps_2d, indexed_orientation, indexed_dims) in generator:
    for (indexed_img_tensor, indexed_ori_img, indexed_label, indexed_pair) in generator:
        # print(indexed_orientation)
        temp_img = indexed_img_tensor
        temp_img_ori = np.array(indexed_ori_img)
        
        # print(temp_img_ori.shape)
        
        cam = Camera(use_own = config_cam['use_own'], img=temp_img_ori, distortion_coef=config_cam['distortion_coef'], fx=config_cam['fx'], fy=config_cam['fy'], cx=config_cam['cx'], cy=config_cam['cy'])
        
        
        curr_img = np.array(indexed_ori_img[0])
        # print(curr_img.shape)
        # print(curr_img)
        
        if my_vpd.remove_fisheye:
            cam = Camera(use_own=False, img=curr_img, distortion_coef=config['Camera']['distortion_coef'],
                            fx=config['Camera']['fx'], fy=config['Camera']['fy'], cx=config['Camera']['cx'], cy=config['Camera']['cy'])
            curr_img = cam.remove_fisheye(curr_img)
        else:
            cam = None

        vps_3d = my_vpd.get_vanishing_points(curr_img)
        vps_2d = my_vpd.get_vanishing_points_2d(curr_img)

        vps_2d, vps_3d = my_vpd.get_vp_loc(curr_img, vps_2d, vps_3d)
        vp1, vp2, vp3 = vps_2d
        print(vp1, vp2, vp3)
        my_vpd.draw_grid(curr_img, vp1, vp2, vp3)
        
        plt.imshow(curr_img)
        plt.show()
        # print(count)
        count += 1
           
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)


