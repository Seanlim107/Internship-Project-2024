from lib.Dataset import ConstructionDataset
from lib.utils import parse_yaml, get_angle, save_checkpoint, load_checkpoint
from lib.VP_Detector import VPD
from torchvision.models import vgg, alexnet, efficientnet_b0
from my_model.Model import BB_Guesser_Model
from lib.loss import Custom_Loss_v2
import torch
import os
import cv2
import re
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils import data
import numpy as np
from lib.train import train_loop, evaluate
import wandb

if torch.cuda.is_available():
  torch.cuda.empty_cache()
  device = torch.device('cuda') 
else: 
  device = torch.device('cpu')
print(device)

####################################################################################################################################################################################################
# The following code is used to train the OriDim model based on configuration parameters in config.yaml
# Note: Code is not available to be run at the moment due to the ACSD dataset being prohibited to be distributed as per company restrictions on data sensitivity 
####################################################################################################################################################################################################


def main(config):
    #Initialization
    filedir = os.path.dirname(__file__)

   # Initialise dataset variables
    config_dataset = config['Dataset']
    config_cons = config_dataset['Construction']
    batch_size = config_dataset['batch_size'] # Only for testing purposes
    backbone_name = config['Models']['3d_guesser_backbone']
    enable_wandb = config['General']['enable_wandb']
    seed = config['General']['seed']
    num_epochs = config['Models']['epochs']
    set_sizes = [config_cons['train_size'], config_cons['test_size'], config_cons['valid_size']]
    classification_set = ['classification', 'regression']
    is_classification = config['Models']['3d_guesser_classification']
    classification = classification_set[is_classification]
    
    # Load up ACSD dataset
    consDataset = ConstructionDataset(config, crops=True)
    params = {'batch_size': batch_size,
                'shuffle':True,
                'num_workers': 6}
    
    # Set random seed
    if(seed is not None):
      torch.manual_seed(seed)
    gen_seed = torch.Generator().manual_seed(seed)
    
    # Split dataset
    train_data, test_data, valid_data = random_split(consDataset, set_sizes, generator=gen_seed)
    train_generator = data.DataLoader(train_data, **params)
    test_generator = data.DataLoader(test_data, **params)
    valid_generator = data.DataLoader(valid_data, **params)
    num_angles = config['Models']['3d_guesser_proposals']
    
    # Set wandb
    if(enable_wandb):
      if not is_classification:
        run = wandb.init(project='Construction_3D', name=f'logger-3Dim_1Orient_{backbone_name}_{classification}_{num_angles}bins')
      else:
        run = wandb.init(project='Construction_3D', name=f'logger-3Dim_1Orient_{backbone_name}_{classification}')
    
    resize_size = config['Dataset']['Construction']['resize_size']
    crop_size = config['Dataset']['Construction']['crop_size']
    model = BB_Guesser_Model(config, backbone_name=backbone_name, angles=num_angles, proposals=1, resize_size=crop_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = Custom_Loss_v2(config)
    
    if (is_classification):
      ori_name = f'3D_Guesser_Train_{backbone_name}_classification_{num_angles}bins'
    else:
      ori_name = f'3D_Guesser_Train_{backbone_name}_regression'
          
    r = re.compile(f'^{ori_name}')

    # Load checkpoints
    checkpointdir = os.path.join(filedir, 'checkpoints')
    list_ckpts = os.listdir(checkpointdir)
    newlist = list(filter(r.match, list_ckpts))
    
    if len(newlist) > 0:
      newlist = [ckpt.split('_')[-1] for ckpt in newlist]
      newlist_num = str(max([int(ckpt.split('.pth')[0]) for ckpt in newlist])) 
      best=f'{ori_name}_ckpt_{newlist_num}.pth' if len(newlist)>0 else None
      best_path = os.path.join(checkpointdir, best)
      print(f'Checkpoint detected for {best}, Loading Checkpoint')
      best_loss, ckpt_epoch = load_checkpoint(model, best_path, optimizer)
      print(f'Checkpoint loaded, best_loss={best_loss}, starting from epoch {ckpt_epoch}')
    else:
      best_loss = float('inf')
      print('No file deteced, starting from scratch')
      ckpt_epoch=0
    
    model=model.to(device)
    
    for epoch in range(ckpt_epoch, num_epochs):
      train_epoch_avg_loss, train_epoch_avg_loss_ang, train_epoch_avg_loss_dim = train_loop(epoch, train_generator, model, loss_fn, device, optimizer, num_bins=num_angles, num_proposals=1, batch_size=batch_size, classification=is_classification)
      test_epoch_avg_loss, test_epoch_avg_loss_ang, test_epoch_avg_loss_dim = evaluate(epoch, test_generator, model, loss_fn, device, num_bins=num_angles, num_proposals=1, batch_size=batch_size, classification=is_classification)
      valid_epoch_avg_loss, valid_epoch_avg_loss_ang, valid_epoch_avg_loss_dim = evaluate(epoch, valid_generator, model, loss_fn, device, num_bins=num_angles, num_proposals=1, batch_size=batch_size, classification=is_classification)
    
      if(enable_wandb):
        run.log({"Train Loss": train_epoch_avg_loss})
        run.log({"Train Angle Loss": train_epoch_avg_loss_ang})
        run.log({"Train Dimension Loss": train_epoch_avg_loss_dim})
        run.log({"Test Loss": test_epoch_avg_loss})
        run.log({"Test Angle Loss": test_epoch_avg_loss_ang})
        run.log({"Test Dimension Loss": test_epoch_avg_loss_dim})
        run.log({'Valid Loss': valid_epoch_avg_loss})
        run.log({'Valid Angle Loss': valid_epoch_avg_loss_ang})
        run.log({'Valid Dimension Loss': valid_epoch_avg_loss_dim})
        
      print(f'Epoch {epoch}, Average Train Loss = {train_epoch_avg_loss}, Avg Test Loss = {test_epoch_avg_loss}, Avg Valid Loss = {valid_epoch_avg_loss}')
      if test_epoch_avg_loss < best_loss:
        save_checkpoint(epoch, model, ori_name, optimizer, train_epoch_avg_loss)
        best_loss = test_epoch_avg_loss
        print(f'Saving Checkpoint at epoch {epoch}')

        
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)