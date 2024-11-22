import numpy as np
from lib.utils import save_checkpoint
import torch
from tqdm import tqdm
import wandb

def train_loop(epoch, generator, model, loss_fn, device, optimizer, num_bins=16, num_proposals=1, batch_size=1, classification=False):
      # with tqdm(generator, unit="batch") as tepoch:
        epoch_total_loss=0
        epoch_ang_total_loss=0
        epoch_dim_total_loss=0
        i=0
        # for (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, vps_3d, vps_2d, indexed_orientation, indexed_dims) in tepoch:
        for (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, vps_3d, vps_2d, indexed_orientation, indexed_dims) in generator:
          # tepoch.set_description(f"Epoch {epoch}")
          i+=1
          # curr_image = indexed_img_tensor.to(device)
          indexed_orientation = indexed_orientation.to(device)
          curr_crop = indexed_crop_tensor.to(device)
          indexed_dims = indexed_dims.to(device)
          
          pred_orient, pred_dims = model(curr_crop)
          # pwrint(pred_dims.shape)
          if(classification):
            pred_orient=pred_orient.view(-1, num_bins)
            indexed_orientation=indexed_orientation.squeeze(-1).long().view(len(pred_orient))
          else:
            pred_orient=pred_orient.view(-1, 1)
            indexed_orientation=indexed_orientation.squeeze(-1).float().view(len(pred_orient))
          # print(pred_orient.shape)
          # print(pred_dims.shape)
          # print(indexed_orientation.shape)
          # print(pred_dims)
          # print(pred_orient)
          # print(indexed_dims)
          # print(indexed_orientation)
          # print(pred_orient.shape, pred_dims.shape)
          # print(indexed_orientation.shape, indexed_dims.shape)
          dim_loss, ang_loss, total_loss = loss_fn(pred_orient, pred_dims, indexed_orientation, indexed_dims)
          dim_loss=dim_loss.detach()
          ang_loss=ang_loss.detach()
          # dim_loss = dim_loss.to(device)
          # ang_loss = ang_loss.to(device)
          # total_loss = total_loss.to(device)
          # plot_image = np.array(indexed_ori_img.squeeze())
          # plot_crop = np.array(indexed_crop.squeeze())
          
          
          total_loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          
          curr_crop = curr_crop.detach()
          indexed_dims = indexed_dims.detach()
          indexed_orientation = indexed_orientation.detach()
          
          epoch_total_loss+=total_loss.item()
          epoch_ang_total_loss+=ang_loss.item()
          epoch_dim_total_loss+=dim_loss.item()
          
          # tepoch.set_postfix(loss=total_loss.item())
          
          if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        epoch_ang_avg_loss = epoch_ang_total_loss/len(generator)
        epoch_dim_avg_loss = epoch_dim_total_loss/len(generator)
        epoch_avg_loss = epoch_total_loss/len(generator)
        return epoch_avg_loss, epoch_ang_avg_loss, epoch_dim_avg_loss
      
def evaluate(epoch, generator, model, loss_fn, device, num_bins=16, num_proposals=1, batch_size=1, classification=False):
      # with tqdm(generator, unit="batch") as tepoch:
        with torch.no_grad():
          epoch_total_loss=0
          epoch_dim_total_loss=0
          epoch_ang_total_loss=0
          # for (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, vps_3d, vps_2d, indexed_orientation, indexed_dims) in tepoch:
          for (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, vps_3d, vps_2d, indexed_orientation, indexed_dims) in generator:
            # tepoch.set_description(f"Epoch {epoch}")
            # curr_image = indexed_img_tensor.to(device)
            indexed_orientation = indexed_orientation.to(device)
            curr_crop = indexed_crop_tensor.to(device)
            indexed_dims = indexed_dims.to(device)
            pred_orient, pred_dims = model(curr_crop)
            if(classification):
              pred_orient=pred_orient.view(-1, num_bins)
              
              indexed_orientation=indexed_orientation.squeeze(-1).long().view(len(pred_orient))
              
            else:
              pred_orient=pred_orient.view(-1, num_bins)
              indexed_orientation=indexed_orientation.squeeze(-1).float().view(len(pred_orient))
            # print(pred_dims)
            # print(pred_orient)
            # print(indexed_dims)
            # print(indexed_orientation)
            dim_loss, ang_loss, total_loss = loss_fn(pred_orient, pred_dims, indexed_orientation, indexed_dims)
            # dim_loss = dim_loss.to(device)
            # ang_loss = ang_loss.to(device)
            # total_loss = total_loss.to(device)
            # plot_image = np.array(indexed_ori_img.squeeze())
            # plot_crop = np.array(indexed_crop.squeeze())
            epoch_total_loss+=total_loss.item()
            epoch_ang_total_loss+=ang_loss.item()
            epoch_dim_total_loss+=dim_loss.item()
        
        epoch_ang_avg_loss = epoch_ang_total_loss/len(generator)
        epoch_dim_avg_loss = epoch_dim_total_loss/len(generator)
        epoch_avg_loss = epoch_total_loss/len(generator)
        return epoch_avg_loss, epoch_ang_avg_loss, epoch_dim_avg_loss
      