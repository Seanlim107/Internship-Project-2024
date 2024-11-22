from lib.Validation_Dataset import ValidationDataset
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
# import tensorflow as tf
# import torchvision.models as models
# import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(config):
    print(device)
    #Initialization
    filedir = os.path.dirname(__file__)
    best_loss = float('inf')
    safe_distancing = config['General']['safe_dist']
    debug = config['General']['debug']
    use_scale = config['General']['use_scale']
    use_cam = config['General']['use_cam']
   # Initialise dataset stuff
    batch_size = 1 # Only for testing purposes
    danger_levels = [safe_distancing/3*(i+1) for i in range(3)]
    seed = config['General']['seed']
    crop_size = config['Dataset']['Construction']['crop_size']
    resize_size = config['Dataset']['Construction']['resize_size']
    consDataset = ValidationDataset(config, crops=False)
    clip_boxes = config['General']['clip_boxes']
    # consDataset = ConstructionDataset(config, crops=True)

    classification = config['Models']['3d_guesser_classification']
    num_angles = config['Models']['3d_guesser_proposals']
    params = {'batch_size': batch_size,
                'shuffle':True,
                'num_workers': 6}
    if(seed is not None):
      torch.manual_seed(config['General']['seed'])
    generator = data.DataLoader(consDataset, **params)
    # generator = consDataset
    
    # Initialise trainng stuff
    # weightpath = os.path.join(filedir, 'runs/detect/train6/weights/best.pt')
    weightpath = os.path.join(filedir, 'runs/validation/train/weights/best.pt')
    yolo = YOLO(weightpath)
    # yolo = YOLOv10(weightpath)
    # yolo.eval()
    
    # backbone = vgg.vgg19_bn(weights='IMAGENET1K_V1')
    # backbone = efficientnet_b0(weights='IMAGENET1K_V1')
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
    # file_name = f'{ori_name}_ckpt.pth'
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
    # for (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, new_vps_3d, new_vps_2d, indexed_orientation, indexed_dims) in generator:
    for (indexed_img_tensor, indexed_ori_img, indexed_img_path) in generator:
      img_path = indexed_img_path
      # print(indexed_orientation)
      cam = Camera(use_own = config_cam['use_own'], img=indexed_ori_img, distortion_coef=config_cam['distortion_coef'], fx=config_cam['fx'], fy=config_cam['fy'], cx=config_cam['cx'], cy=config_cam['cy'])
      flag=0
      starttime = time.time()
      distance = float('inf')
      plot_image = np.array(indexed_ori_img.squeeze())
      list_workers2 = []
      list_nonworkers2 = []
      # curr_image = indexed_img_tensor.to(device)
      # indexed_orientation = indexed_orientation.to(device)
      # curr_crop = indexed_crop_tensor.to(device)
      # indexed_dims = indexed_dims.to(device)
      
      
      with torch.no_grad():
        detections = yolo(img_path, imgsz=resize_size, agnostic_nms=True)
        # detections = yolo(indexed_img_tensor, imgsz=resize_size)
      # img = np.copy(plot_image)
      # print(detections)
      calib_file = "calib_cam_to_cam.txt"

      list_boxes_2 = detections[0].boxes.xyxy

      #List of detected classes
      detected_classes_2 = detections[0].boxes.cls
      
    
      # img = detections[0].plot()
      #   print(img.shape)
      img = plot_image
      

      for detection in detections[0]:
        # print(img, detection.boxes.cls, detection.boxes.xyxy, calib_file)
        # print(img.shape, detection.boxes.cls.shape, detection.boxes.xyxy.shape)
        box = detection.boxes.xyxy.squeeze().cpu().int().numpy()
        box_2d = ([(box[0], box[1]), (box[2], box[3])])
        detectedObject = DetectedObject(img, detection.boxes.cls.cpu(), box_2d, calib_file, resize_size=crop_size)
        # try:
        #     detectedObject = DetectedObject(img, detection.boxes.cls.cpu(), box_2d, calib_file, resize_size)
        # except:
        #     continue
        
        theta_ray = 0
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        box_2d = box_2d
        
        

        # detected_class = detection.boxes.cls
        input_tensor = input_img.to(device).unsqueeze(0)
        # input_tensor = torch.zeros([1,3,224,224]).to(device)
        # input_tensor[0,:,:,:] = input_img
        
        # plt.subplot(1,2,1)
        # plot_indexed_crop_tensor = indexed_crop_tensor.squeeze()*255
        # plot_input_img = input_img.squeeze()*255
        # plt.imshow(plot_indexed_crop_tensor.int().numpy().transpose(1,2,0))
        # plt.subplot(1,2,2)
        # plt.imshow(plot_input_img.int().numpy().transpose(1,2,0))
        # plt.show()
        
        
        [orient, dim] = model(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        if(classification):
          angle = bins[np.argmax(orient, axis=0)]
        else:
          orient = orient[0,:]
          angle = orient
        # print(orient)
        # conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]
        
        

        # angle = orient
        
        cos = np.cos(angle[0])
        sin = np.sin(angle[0])
        alpha = np.arctan2(sin, cos)
        # print(alpha)
        # alpha = orient[0]
        beta = 0
        # beta = orient[1]
        # alpha += angle_bins[argmax]
        # alpha -= np.pi
        # print(alpha)
        
        # print(box_2d)
        detect_class = detection.boxes.cls.cpu().int().numpy()[0]
        dim = np.array(lookup[detect_class]['dimensions'])
        dim = np.flip(dim)
        # print(dim)
        
        location, corners_3d = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, beta=beta, clip=clip_boxes)
        
        # print(location)
        
        detect_label = lookup[detect_class]['name']
        detect_conf = detection.boxes.conf.cpu().float().numpy()[0]
        
        corners_3d = np.array(corners_3d)
        min_x_3d, min_y_3d = np.min(corners_3d, axis=0)
        max_x_3d, max_y_3d = np.max(corners_3d, axis=0)
        box3d = [min_x_3d, min_y_3d, max_x_3d, max_y_3d]
        # print(a)
        # plt.scatter(a[:,0], a[:,1])
        # plt.show()
        # print(detect_class)
        if 0 in detected_classes_2 and any(num != 0 for num in detected_classes_2):
          if(detect_class == 0):
              list_workers2.append((detect_conf, location, detect_class, box, dim, corners_3d, alpha, box_2d, box3d))
          else:
              list_nonworkers2.append((detect_conf, location, detect_class, box, dim, corners_3d, alpha, box_2d, box3d))
        # print(f'Estimated pose:{detect_label} {detect_conf:.2f} {location}')
        # numpy_vertical = np.concatenate((plot_image, img), axis=0)
      
      


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
                # length= estimate_distance_2d(worker[2], nonworker[2])
                # print(length)
                hazard = lookup[class_non]['name']

                if use_scale:     
                  worker_3d_coords = dim_worker / scale
                  distance = estimate_distance_2d(box_worker, box_non) * scale
                elif use_cam:
                  worker_3d_coords = coord_worker
                  hazard_3d_coords = coord_non
                  
                  distance = estimate_distance_centers_3d(worker_3d_coords, hazard_3d_coords, dim)
                
                
                # print(f'3D Coordinates of worker: ,{worker[3]:.2f},{worker_3d_coords}')
                # print(f'3D Coordinates of , {hazard}: , {nonworker[3]:.2f}, {hazard_3d_coords}')
                # print(f'Distance = {distance:.2f}m')
                if(distance < safe_distancing):
                  mode = np.digitize(distance, danger_levels, right=True)
                  # print(f'Unsafe distancing between worker:{conf_worker:.2f} and {hazard}:{conf_non:.2f}, Distance={distance:.2f}m')
                  # print('3D Coordinates of worker: ',conf_worker,worker_3d_coords)
                  # print('3D Coordinates of ', hazard,': ', conf_non, hazard_3d_coords)
                  flag=1
                  refine_regressed_3d_bbox(img, proj_matrix, box_2d_worker, dim_worker, alpha_worker, theta_ray, coord_worker, mode=mode, clip=clip_boxes, beta=beta)
                  refine_regressed_3d_bbox(img, proj_matrix, box_2d_non, dim_non, alpha_non, theta_ray, coord_non, mode=mode, clip=clip_boxes, beta=beta)
                
                
                # draw_lines(box_3d_worker, box_3d_non, img, distance, debug=debug, safe_distancing=safe_distancing)

      time_set += time.time() - starttime
      print(f'Inference time : {time.time() - starttime}')
      # fig = plt.figure(figsize=(9, 8), dpi=80)
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
     
    average_time = time_set/len(generator)
    print(average_time)
        
        
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)