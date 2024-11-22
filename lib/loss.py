from lib.utils import parse_yaml, get_angle
import torch
import os,cv2
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils import data
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


class Custom_Loss_v2(nn.Module):
  def __init__(self, config):
    super(Custom_Loss_v2, self).__init__()
    self.classification = config['Models']['3d_guesser_classification']
    
  def forward(self, orient_pred, dim_pred, orient_real, dim_real, loss_fn=F.mse_loss):
      if not self.classification:
        ang_loss=  loss_fn(orient_pred, orient_real)
        dim_loss=   loss_fn(dim_pred, dim_real)
      else:
        ang_loss=  F.cross_entropy(orient_pred, orient_real)
        dim_loss = loss_fn(dim_pred, dim_real)
        
      total_loss = dim_loss+ang_loss
      return dim_loss, ang_loss, total_loss
