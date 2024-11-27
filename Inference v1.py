# Load a pre-trained YOLOv10n model
# from ultralytics import YOLOv10
from ultralytics import YOLO
from lib.Dataset import ConstructionDataset
from lib.Validation_Dataset import ValidationDataset
from lib.Camera import Camera
from lib.utils import parse_yaml, estimate_distance_2d, draw_lines, estimate_distance_centers_3d
import torch
import os
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2

####################################################################################################################################################################################################
# The following code is used for testing the traditional methods used for distance measurement algorithm
# Creates 2D bounding box using YOLO, measures height of workers, gets scale, then calculates real world distance relative to pixel length of bounding boxes
####################################################################################################################################################################################################

def main(config):
    #Initialization
    flag=0
    filedir = os.path.dirname(__file__)

    # Alternative weight path
    # weightpath = os.path.join(filedir, 'runs/detect/train3/weights/best.pt')
    
    weightpath = os.path.join(filedir, 'runs/validation/train/weights/best.pt')
    
    model = YOLO(weightpath)
    lookup = config['Classes']
    config_cam = config['Camera']
    resize_size = config['Dataset']['Construction']['resize_size']
    safe_distancing = config['General']['safe_dist']
    
    
    # Picking between ACSD dataset or Validation dataset
    # consDataset = ConstructionDataset(config, crops=False)
    consDataset = ValidationDataset(config, crops=False)
    

    generator = consDataset
    
    for (indexed_img_tensor, indexed_ori_img, indexed_img_path) in generator:
        temp_img = indexed_img_tensor
        temp_img_ori = np.array(indexed_ori_img)
        
        cam = Camera(use_own = config_cam['use_own'], img=temp_img_ori, distortion_coef=config_cam['distortion_coef'], fx=config_cam['fx'], fy=config_cam['fy'], cx=config_cam['cx'], cy=config_cam['cy'])
        
        
        with torch.no_grad():
            results = model(indexed_img_path, imgsz=resize_size, agnostic_nms=True)

            list_boxes = results[0].boxes.xyxy
            
            #List of detected classes
            detected_classes = results[0].boxes.cls
            
            
        # Plot original Image with Yolo Detection    
        img = results[0].plot()
        
        
        #Draw lines and send warning if a distance is lower than safe distance
        if 0 in detected_classes and any(num != 0 for num in detected_classes):
            ##################################################################################################################################################
            # List workers and List Non Workers have the following:
            # (0: index wrt detected objects in YOLO accordingly, 1: Class in int format, 2: xyxy, 3: Confidence Score for ease of labelling when plotting)
            ##################################################################################################################################################
            list_conf = results[0].boxes.conf
            list_workers = [(i, detected_classes[i].cpu().item(), list_boxes[i].cpu().data, list_conf[i].cpu().item()) for i in range(len(detected_classes)) if int(detected_classes[i]) == 0]
            list_nonworkers = [(i, detected_classes[i].cpu().item(), list_boxes[i].cpu().data, list_conf[i].cpu().item()) for i in range(len(detected_classes)) if int(detected_classes[i]) != 0]
            
            # Calculate distance between exactly 1 worker and exactly 1 non worker (hazards)
            for worker in list_workers:
                for nonworker in list_nonworkers:
                    length= estimate_distance_2d(worker[2], nonworker[2])
                    hazard = lookup[nonworker[1]]['name']
                    worker_dim = lookup[worker[1]]['dimensions']
                    hazard_dim = lookup[nonworker[1]]['dimensions']
                    
                    scale = cam.get_scale(worker[2], worker_dim)
                    
                    worker_3d_coords = cam.find_real_coords(worker[2], worker_dim)
                    hazard_3d_coords = cam.find_real_coords(nonworker[2], hazard_dim)
                    distance = scale*length

                    print(f'Worker: {worker[3]:.2f},{hazard}:{nonworker[3]:.2f}, Distance = {distance}m')
                    
                    if(distance < safe_distancing):
                        print(f'Unsafe distancing between worker:{worker[3]:.2f} and {hazard}:{nonworker[3]:.2f}, Distance={distance:.2f}')
                        draw_lines(worker[2], nonworker[2], img, dist=distance, debug=False, safe_distancing = safe_distancing)
                        flag=1
                        print('_______________________________________________________________________________________________________________________________________________')
        else:
            print('No workers/hazards detected')
       
       # Displa results with button delay
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show(block=False)
        keyboardClick=False
        while keyboardClick != True:
            keyboardClick=plt.waitforbuttonpress()
        if(flag): 
            keyboardClick=False
            while keyboardClick != True:
                keyboardClick=plt.waitforbuttonpress()
        else:
            plt.pause(0.5)
        plt.clf()
           
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)


