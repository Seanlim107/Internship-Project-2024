from lib.Dataset import ConstructionDataset
from lib.utils import parse_yaml, get_angle, save_checkpoint, load_checkpoint, draw_lines, estimate_distance_centers_3d, estimate_distance_2d
from lib.VP_Detector import VPD
from lib.Camera import Camera
from my_model.Model import BB_Guesser_Model
from ultralytics import YOLO
import torch
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils import data
import time
from lib.utils import create_bins, get_bin

import numpy as np
from lib_3D.Dataset import *
from lib_3D.Plotting import *
import re

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

####################################################################################################################################################################################################
# Note: This code is currently not available to be run due to requiring the ACSD dataset which is not provided as per company restrictions with sensitive data
# The following code is used for testing OriDim on ACSD dataset
# Creates 2D bounding box using YOLO, cropped images from YOLO's bounding box is fed into OriDim to guess dimensions and orientation
####################################################################################################################################################################################################


def main(config):
    #Initialization
    filedir = os.path.dirname(__file__)
    best_loss = float('inf')
    safe_distancing = config['General']['safe_dist']
    debug = config['General']['debug']
    use_scale = config['General']['use_scale']
    use_cam = config['General']['use_cam']
    
   # Initialise dataset variables
    batch_size = 1 
    danger_levels = [safe_distancing/3*(i+1) for i in range(3)]
    seed = config['General']['seed']
    crop_size = config['Dataset']['Construction']['crop_size']
    resize_size = config['Dataset']['Construction']['resize_size']
    consDataset = ConstructionDataset(config, crops=False)
    clip_boxes = config['General']['clip_boxes']

    classification = config['Models']['3d_guesser_classification']
    num_angles = config['Models']['3d_guesser_proposals']
    params = {'batch_size': batch_size,
                'shuffle':True,
                'num_workers': 6}
    if(seed is not None):
      torch.manual_seed(config['General']['seed'])
    generator = data.DataLoader(consDataset, **params)

    # Initialise YOLO variables
    weightpath = os.path.join(filedir, 'runs/detect/train3/weights/best.pt')
    yolo = YOLO(weightpath)
    yolo.eval()

    # Initialise backbone
    backbone_name = config['Models']['3d_guesser_backbone']
    model = BB_Guesser_Model(config, backbone_name=backbone_name, angles=num_angles, proposals=1, resize_size=crop_size)
    model.eval()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    if (classification):
      ori_name = f'3D_Guesser_Train_{backbone_name}_classification_{num_angles}bins'
      bins = create_bins(num_bins=num_angles)
    else:
      ori_name = f'3D_Guesser_Train_{backbone_name}_regression'

    # Filter for latest checkpoint depending on configuration
    r = re.compile(f'^{ori_name}')
    checkpointdir = os.path.join(filedir, 'checkpoints')
    list_ckpts = os.listdir(checkpointdir)
    newlist = list(filter(r.match, list_ckpts))
    config_cam = config['Camera']
    
    if len(newlist) > 0:
      newlist = [ckpt.split('_')[-1] for ckpt in newlist]
      newlist_num = str(max([int(ckpt.split('.pth')[0]) for ckpt in newlist])) 
      best=f'{ori_name}_ckpt_{newlist_num}.pth' if len(newlist)>0 else None
      best_path = os.path.join(checkpointdir, best)
      print(f'Checkpoint detected for {best}, Loading Checkpoint')
      best_loss, ckpt_epoch = load_checkpoint(model, best_path, optimizer)
    else:
      raise Exception('No checkpoint detected')
    
    my_vpd = VPD(config)
    lookup = config['Classes']
    time_set = 0 
    
    
    for (indexed_img_tensor, indexed_ori_img, indexed_label, indexed_pair) in generator:
      img_path, label_path = indexed_pair
      cam = Camera(use_own = config_cam['use_own'], img=indexed_ori_img, distortion_coef=config_cam['distortion_coef'], fx=config_cam['fx'], fy=config_cam['fy'], cx=config_cam['cx'], cy=config_cam['cy'])
      flag=0
      starttime = time.time()
      distance = float('inf')
      plot_image = np.array(indexed_ori_img.squeeze())
      list_workers2 = []
      list_nonworkers2 = []

      
      # Loads camera matrix
      with torch.no_grad():
        detections = yolo(img_path, imgsz=resize_size, agnostic_nms=True)
      calib_file = "calib_cam_to_cam.txt"

      list_boxes_2 = detections[0].boxes.xyxy

      #List of detected classes
      detected_classes_2 = detections[0].boxes.cls
      
      img = plot_image
      
      # Function for guessing orientation and dimension + plotting 3D box
      # Uses cropped image to get orientation and dimension using OriDim
      # Plot 3D box using projection matrix, dimension, orientation
      for detection in detections[0]:
        box = detection.boxes.xyxy.squeeze().cpu().int().numpy()
        box_2d = ([(box[0], box[1]), (box[2], box[3])])
        detectedObject = DetectedObject(img, detection.boxes.cls.cpu(), box_2d, calib_file, resize_size=crop_size)
        
        theta_ray = 0
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        box_2d = box_2d
        
        input_tensor = input_img.to(device).unsqueeze(0)
        
        [orient, dim] = model(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        if(classification):
          angle = bins[np.argmax(orient, axis=0)]
        else:
          orient = orient[0,:]
          angle = orient

        dim = dim.cpu().data.numpy()[0, :]
        
        cos = np.cos(angle[0])
        sin = np.sin(angle[0])
        alpha = np.arctan2(sin, cos)

        beta = 0

        detect_class = detection.boxes.cls.cpu().int().numpy()[0]
        dim = np.array(lookup[detect_class]['dimensions'])
        dim = np.flip(dim)

        location, corners_3d = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, beta=beta, clip=clip_boxes)
        
        detect_label = lookup[detect_class]['name']
        detect_conf = detection.boxes.conf.cpu().float().numpy()[0]
        
        corners_3d = np.array(corners_3d)
        min_x_3d, min_y_3d = np.min(corners_3d, axis=0)
        max_x_3d, max_y_3d = np.max(corners_3d, axis=0)
        box3d = [min_x_3d, min_y_3d, max_x_3d, max_y_3d]

        if 0 in detected_classes_2 and any(num != 0 for num in detected_classes_2):
          if(detect_class == 0):
              list_workers2.append((detect_conf, location, detect_class, box, dim, corners_3d, alpha, box_2d, box3d))
          else:
              list_nonworkers2.append((detect_conf, location, detect_class, box, dim, corners_3d, alpha, box_2d, box3d))
      

      # Compares exactly 1 worker and 1 non worker for each iteration
      if(len(list_nonworkers2) > 0 and len(list_workers2) > 0):
          if(use_scale):
            scale_total = 0
            for conf_worker, coord_worker, class_worker, box_worker, dim_worker, corners_3d_worker, alpha_worker, box_2d_worker, box_3d_worker in list_workers2:
              temp_scale = cam.get_scale(box_worker, dim_worker)
              scale_total+=temp_scale
            scale = scale_total/len(list_workers2)
          for worker in list_workers2:
            for nonworker in list_nonworkers2:
                conf_worker, coord_worker, class_worker, box_worker, dim_worker, corners_3d_worker, alpha_worker, box_2d_worker, box_3d_worker = worker
                conf_non, coord_non, class_non, box_non, dim_non, corners_3d_non, alpha_non, box_2d_non, box_3d_non = nonworker

                hazard = lookup[class_non]['name']

                # Convert 3D dimensions depending on variation
                if use_scale:     
                  worker_3d_coords = dim_worker / scale
                  distance = estimate_distance_2d(box_worker, box_non) * scale
                elif use_cam:
                  worker_3d_coords = coord_worker
                  hazard_3d_coords = coord_non
                  
                  distance = estimate_distance_centers_3d(worker_3d_coords, hazard_3d_coords, dim)

                # Set alert for non safe distancing
                if(distance < safe_distancing):
                  mode = np.digitize(distance, danger_levels, right=True)

                  flag=1
                  refine_regressed_3d_bbox(img, proj_matrix, box_2d_worker, dim_worker, alpha_worker, theta_ray, coord_worker, mode=mode, clip=clip_boxes, beta=beta)
                  refine_regressed_3d_bbox(img, proj_matrix, box_2d_non, dim_non, alpha_non, theta_ray, coord_non, mode=mode, clip=clip_boxes, beta=beta)

      # Calculate inference time
      time_set += time.time() - starttime
      print(f'Inference time : {time.time() - starttime}')

      # Display Image
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      plt.imshow(img)
      plt.show(block=False)
      if(flag):
        keyboardClick=False
        while keyboardClick != True:
            keyboardClick=plt.waitforbuttonpress()
      else:
        plt.pause(0.5)
      plt.clf()
     
    average_time = time_set/len(generator)
    print(average_time)
        
        
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)