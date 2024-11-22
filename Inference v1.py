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

def main(config):
    #Initialization
    flag=0
    filedir = os.path.dirname(__file__)
    # weightpath = os.path.join(filedir, 'runs/detect/train12/weights/best.pt')
    # model = YOLOv10(weightpath)
    # weightpath = os.path.join(filedir, 'runs/detect/train3/weights/best.pt')
    weightpath = os.path.join(filedir, 'runs/validation/train/weights/best.pt')
    model = YOLO(weightpath)
    lookup = config['Classes']
    config_cam = config['Camera']
    resize_size = config['Dataset']['Construction']['resize_size']
    batch_size = 1 # Only for testing purposes
    safe_distancing = config['General']['safe_dist']
    
    
    # consDataset = ConstructionDataset(config, crops=False)
    consDataset = ValidationDataset(config, crops=False)
    

    generator = consDataset
    
    # for (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, new_vps_3d, new_vps_2d, indexed_orientation, indexed_dims) in generator:
    for (indexed_img_tensor, indexed_ori_img, indexed_img_path) in generator:
    # for (indexed_img_tensor, indexed_ori_img, indexed_label, indexed_pair) in generator:
        temp_img = indexed_img_tensor
        temp_img_ori = np.array(indexed_ori_img)
        # img_path, label_path = indexed_pair
        # print(temp_img_ori.shape)
        
        cam = Camera(use_own = config_cam['use_own'], img=temp_img_ori, distortion_coef=config_cam['distortion_coef'], fx=config_cam['fx'], fy=config_cam['fy'], cx=config_cam['cx'], cy=config_cam['cy'])
        
        
        with torch.no_grad():
            results = model(indexed_img_path, imgsz=resize_size, agnostic_nms=True)

            #Predicted coordinates of box (top left, bottom right)
            list_boxes = results[0].boxes.xyxy
            
            # print('xywh', results[0].boxes.xywh)
            
            #List of detected classes
            detected_classes = results[0].boxes.cls
            
            
        # Plot original Image with Yolo Detection    
        img = results[0].plot()
        
        #Draw lines and send warning if a distance is lower than safe distance
        if 0 in detected_classes and any(num != 0 for num in detected_classes):
            list_conf = results[0].boxes.conf
            list_workers = [(i, detected_classes[i].cpu().item(), list_boxes[i].cpu().data, list_conf[i].cpu().item()) for i in range(len(detected_classes)) if int(detected_classes[i]) == 0]
            list_nonworkers = [(i, detected_classes[i].cpu().item(), list_boxes[i].cpu().data, list_conf[i].cpu().item()) for i in range(len(detected_classes)) if int(detected_classes[i]) != 0]
            
            ##################################################################################################################################################
            # List workers and List Non Workers have the following:
            # (0: index wrt detected objects in YOLO accordingly, 1: Class in int format, 2: xyxy, 3: Confidence Score for ease of labelling when plotting)
            ##################################################################################################################################################
            for worker in list_workers:
                for nonworker in list_nonworkers:
                    length= estimate_distance_2d(worker[2], nonworker[2])
                    # print(length)
                    hazard = lookup[nonworker[1]]['name']
                    worker_dim = lookup[worker[1]]['dimensions']
                    hazard_dim = lookup[nonworker[1]]['dimensions']
                    
                    scale = cam.get_scale(worker[2], worker_dim)
                    
                    worker_3d_coords = cam.find_real_coords(worker[2], worker_dim)
                    hazard_3d_coords = cam.find_real_coords(nonworker[2], hazard_dim)
                    distance = scale*length
                    # distance = estimate_distance_centers_3d(worker_3d_coords, hazard_3d_coords)
                    
                    # print('worker: ',worker[3],worker_3d_coords)
                    # print(hazard,': ', nonworker[3], hazard_3d_coords)
                    print(f'Worker: {worker[3]:.2f},{hazard}:{nonworker[3]:.2f}, Distance = {distance}m')
                    
                    if(distance < safe_distancing):
                        print(f'Unsafe distancing between worker:{worker[3]:.2f} and {hazard}:{nonworker[3]:.2f}, Distance={distance:.2f}')
                        draw_lines(worker[2], nonworker[2], img, dist=distance, debug=False, safe_distancing = safe_distancing)
                        flag=1
                        print('_______________________________________________________________________________________________________________________________________________')
        else:
            print('No workers/hazards detected')
       
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
            # if(flag):
            #     keyboardClick=False
            #     while keyboardClick != True:
            #         keyboardClick=plt.waitforbuttonpress() 
            # else:
            #     plt.pause(0.5)
            # plt.clf()
        # cv2.imshow("YOLO Results with Lines", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
           
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)


