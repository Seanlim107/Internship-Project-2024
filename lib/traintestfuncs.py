from Dataset import ConstructionDataset
from Camera import Camera
from utils import parse_yaml, get_angle, save_checkpoint, load_checkpoint
from VP_Detector import VPD
from loss import Custom_Loss
import torch
import os,cv2
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils import data
import numpy as np
import tensorflow as tf
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
     
     
     
def train(model, generator, optimizer, config, camera=None, num_epochs=10):
  loss_fn = Custom_Loss()
  max_loss = float('inf')
  
  crop_size = config['General']['crop_size']
  
  for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    iter=0
    epoch_loss = 0
    for indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, new_vps_3d, new_vps_2d, indexed_orientation, indexed_dims in generator:
      