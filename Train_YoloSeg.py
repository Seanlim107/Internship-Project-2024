from lib.MOCS_Dataset import MOCSDataset
from lib.Camera import Camera
from lib.utils import parse_yaml
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO


############################################################################################################################################################
# Note: This code is currently not available to be run due to requiring the MOCS dataset which is too big to be uploaded in the GitHub repository
# Code for training the YOLO Segmentation model
############################################################################################################################################################


def main(config):
    #Initialization
    model = YOLO("yolov8n-seg.pt")
    results = model.train(data="config_MOCS_yolo.yaml", epochs=config['Models']['epochs'], imgsz=config['Dataset']['MOCS']['resize_size'])
    
    print(results)
           
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)


