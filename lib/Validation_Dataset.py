from torch.utils import data
from torchvision import transforms
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import shutil
import torch
from lib.VP_Detector import VPD
from lib.utils import get_angle, create_bins, get_bin

class ValidationDataset(data.Dataset):
    def __init__(self, config, crops=True):
        self.filedir = os.path.dirname(os.path.dirname(__file__))
        # # print(self.filedir)
        
        # self.images_dir = os.path.join(mainpath, imgdirname)
        # self.labels_dir = os.path.join(mainpath, labeldirname)
        
        config_yoloprep = config['Dataset']['Construction']
        mainpath = 'Validation'
        imgdirname = config_yoloprep['img_dir_name']
        labeldirname = config_yoloprep['lab_dir_name']
        batchdirname = config_yoloprep['Batch_dir_name']
        start_num_batch = config_yoloprep['start_batch_num']
        end_num_batch = config_yoloprep['end_batch_num']
        self.classification = config['Models']['3d_guesser_classification']
        self.num_angles = config['Models']['3d_guesser_proposals']
        
        self.moving_dir = config_yoloprep['moving_dir']
        self.img_ext = config_yoloprep['img_ext']
        self.label_ext = config_yoloprep['label_ext']
        self.train_size = config_yoloprep['train_size']
        self.test_size = config_yoloprep['test_size']
        self.valid_size = config_yoloprep['valid_size']
        self.resize_size = tuple([config['Dataset']['Construction']['resize_size'], config['Dataset']['Construction']['resize_size']])
        self.crop_size = tuple([config['Dataset']['Construction']['crop_size'], config['Dataset']['Construction']['crop_size']])
        seed = config['General']['seed']
        self.crops = crops
        self.lookup = config['Classes']
        
        self.images_dir = os.path.join(self.filedir, mainpath, 'images', 'train')
        
        self.images_paths = []
        random.seed(seed)
        self.obj_list = []
        self.ori_img_list = []
        self.clas_list = []
        
        self.vpd = VPD(config)
        
        assert 0 <= self.train_size <= 1, "Train size must be between 0 and 1"
        assert 0 <= self.test_size <= 1, "Test size must be between 0 and 1"
        assert 0 <= self.valid_size <= 1, "Valid size must be between 0 and 1"
        assert 0 <= self.train_size+self.test_size+self.valid_size <=1, "train size test size and valid size must add up to be between 0 and 1"

        
        self.dict = {  0: "worker",
        1: "suspended load",
        2: "static crane",
        3: "crane",
        4: "roller",
        5: "bulldozer",
        6: "excavator",
        7: "truck",
        8: "loader",
        9: "pump truck",
        10: "concrete mixer",
        11: "pile driving",
        12: "forklift"}

        # List all image paths and label paths
        try:

            for root,_,files in os.walk(self.images_dir):
                for file in files:
                    curr_img_path = os.path.join(self.images_dir, file)
                    self.images_paths.append(curr_img_path)

        except:
            raise Exception('Unexpected Error occured')

        
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
            # indexed_ori_img_path=self.images_paths[index]
            indexed_img   = self.format_img(self.getimage(self.images_paths[index]))
            indexed_ori_img = self.getimage(self.images_paths[index])
            indexed_path = self.images_paths[index]
            
            return indexed_img, indexed_ori_img, indexed_path
        
    def getimage(self, imgpath):
        img = cv2.imread(imgpath)  
        return np.array(img)
    
    
    def format_img(self,img,box_2d=None):
        temp_img = cv2.resize(img, self.resize_size)/255.0
        temp_img = np.transpose(temp_img, (2,0,1))
        temp_img = torch.tensor(temp_img, dtype=torch.float32)

        return temp_img
    
    