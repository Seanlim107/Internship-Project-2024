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


###############################################################################
# The following code is used for accessing the ACSD dataset
# Code is currently not available due to the ACSD dataset not provided in this repository
###############################################################################

class ConstructionDataset(data.Dataset):
    def __init__(self, config, crops=True):
        self.filedir = os.path.dirname(os.path.dirname(__file__))
        
        config_yoloprep = config['Dataset']['Construction']
        mainpath = config_yoloprep['main_dir']
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
        
        self.images_dir = os.path.join(self.filedir, mainpath, imgdirname)
        self.labels_dir = os.path.join(self.filedir, mainpath, labeldirname)
        
        self.images_paths = []
        self.labels_paths = []
        random.seed(seed)
        self.obj_list = []
        self.ori_img_list = []
        self.clas_list = []
        self.pairs=[]
        
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
            for i in range(start_num_batch,end_num_batch+1):
                img_batch_dir = os.path.join(self.images_dir,f'{batchdirname}{i}')
                label_batch_dir = os.path.join(self.labels_dir,f'{batchdirname}{i}')
                
                for root,_,files in os.walk(img_batch_dir):
                    for file in files:
                        curr_img_path = os.path.join(img_batch_dir, file)
                        self.images_paths.append(curr_img_path)
                        
                for root,_,files in os.walk(label_batch_dir):
                    for file in files:
                        curr_label_path = os.path.join(label_batch_dir, file)
                        self.labels_paths.append(curr_label_path)
        except:
            raise Exception('Unexpected Error occured')

        if(len(self.images_paths)!=len(self.labels_paths)):
            raise Exception('Images and Label Sizes are not Same')
        
        for i in range(len(self.images_paths)):
            self.pairs.append((self.images_paths[i], self.labels_paths[i]))
            
        if self.crops:
            self.makelabel()
        
    
    def createYolo(self):
        # Creates directory for Yolo
        self.nameTraining = ["train", "test", "valid"]
        self.nameData = ["images", "labels"]
        self.nameYaml = "data.yaml"
        
        self.create_Dataset()
        self.copy_files()
        
    def __len__(self):
        if self.crops:
            return len(self.obj_list)
        else:
            return len(self.pairs)
    
    def __getitem__(self, index):
        if(self.crops):
            
            indexed_ori_img_path = self.ori_img_list[index]
            indexed_ori_img = self.getimage(indexed_ori_img_path)
            height, width = indexed_ori_img.shape[:2]
            indexed_img_tensor = self.format_img(indexed_ori_img)
            indexed_clas    = self.obj_list[index][0]
            indexed_box2d   = self.obj_list[index][1]
            indexed_crop    = self.getcrop(indexed_ori_img_path, indexed_box2d)
            indexed_crop_tensor = self.format_crop(indexed_crop)

            vps_3d = self.vpd.get_vanishing_points(indexed_ori_img)
            vps_2d = self.vpd.get_vanishing_points_2d(indexed_ori_img)
            
            new_vps_2d, new_vps_3d = self.vpd.get_vp_loc(indexed_ori_img, vps_2d, vps_3d)
            new_vps_2d = new_vps_2d
            v1, v2, v3 = new_vps_2d

            x_cent, y_cent, box_width, box_height = indexed_box2d
            indexed_yaw = get_angle(torch.Tensor([x_cent, y_cent])*torch.Tensor([width,height]), torch.Tensor(v1)) + np.pi
            indexed_roll = torch.Tensor([0])
            indexed_pitch = torch.Tensor([np.pi/4])
            if(self.classification):
                bins = create_bins(self.num_angles)
                indexed_yaw = get_bin(indexed_yaw, bins)

                indexed_orientation = torch.Tensor([indexed_yaw]).unsqueeze(0)
            else:
                indexed_orientation = torch.Tensor([indexed_yaw]).unsqueeze(0)
            indexed_dims = torch.Tensor(self.lookup[indexed_clas]['dimensions'])

            return (indexed_crop, indexed_crop_tensor, indexed_box2d, indexed_clas, indexed_ori_img, indexed_img_tensor, new_vps_3d, new_vps_2d, indexed_orientation, indexed_dims)
        else:
            indexed_label = self.getlabel(self.labels_paths[index])
            indexed_img   = self.format_img(self.getimage(self.images_paths[index]))
            indexed_ori_img = self.getimage(self.images_paths[index])
            indexed_pair = self.pairs[index]
            
            return indexed_img, indexed_ori_img, indexed_label, indexed_pair
        
    def getimage(self, imgpath):
        img = cv2.imread(imgpath)

        return np.array(img)
    
    
    def create_Dataset(self):
        flag=1
        for train_type in self.nameTraining:
            for imlab in self.nameData:
                dir_to_make = os.path.join(self.filedir, self.moving_dir, imlab, train_type)

                if not os.path.exists(dir_to_make):
                    os.makedirs(dir_to_make)
                    flag=0
        
        if flag:
            print("Folders already exist")
        else:
            print("Folder(s) created")
                    
    def copy_files(self):
        # Copies folders of AITIS' format (Images/Batch1) to fit to YOLO format [train/images] or [train/labels]
        
        # Random shuffle
        random.shuffle(self.pairs)
        
        # Split pairs
        len_pairs = len(self.pairs)
        num_train = int(len_pairs * self.train_size)
        num_test = int(len_pairs* self.test_size)
        num_valid = int(len_pairs - num_train - num_test)
        
        train_pairs = self.pairs[:num_train]
        test_pairs = self.pairs[num_train: num_train+num_test]
        valid_pairs = self.pairs[num_train+num_test:]
        
        trainfoldername,testfoldername,validfoldername = self.nameTraining
        imfoldername, lbfoldername = self.nameData
        
        
        for im,lb in train_pairs:
            im_filename = os.path.basename(im)
            lb_filename = os.path.basename(lb)
            shutil.copy(im, os.path.join(self.filedir, self.moving_dir,  imfoldername, trainfoldername, im_filename))
            shutil.copy(lb, os.path.join(self.filedir, self.moving_dir, lbfoldername, trainfoldername, lb_filename))
        for im,lb in test_pairs:
            im_filename = os.path.basename(im)
            lb_filename = os.path.basename(lb)
            shutil.copy(im, os.path.join(self.filedir, self.moving_dir, imfoldername, testfoldername, im_filename))
            shutil.copy(lb, os.path.join(self.filedir, self.moving_dir, lbfoldername, testfoldername, lb_filename))
        for im,lb in valid_pairs:
            im_filename = os.path.basename(im)
            lb_filename = os.path.basename(lb)
            shutil.copy(im, os.path.join(self.filedir, self.moving_dir, imfoldername, validfoldername, im_filename))
            shutil.copy(lb, os.path.join(self.filedir, self.moving_dir, lbfoldername, validfoldername, lb_filename))
        print("Files copied")    

    def format_img(self,img,box_2d=None):
        temp_img = cv2.resize(img, self.resize_size)/255.0
        temp_img = np.transpose(temp_img, (2,0,1))
        temp_img = torch.tensor(temp_img, dtype=torch.float32)

        return temp_img
    
    
    def format_crop(self,img,box_2d=None):
        temp_img = cv2.resize(img, self.crop_size)/255.0
        temp_img = np.transpose(temp_img, (2,0,1))
        temp_img = torch.tensor(temp_img, dtype=torch.float32)

        return temp_img
            
    def getcrop(self, imgpath, box2d):
        img = self.getimage(imgpath)
        size_img=np.shape(img)

        x_center=float(box2d[0])*size_img[1]
        y_center=float(box2d[1])*size_img[0]
        width=float(box2d[2])*size_img[1]
        height=float(box2d[3])*size_img[0]
        x_min = int(x_center-width/2)
        x_max = int(x_center+width/2)
        y_min = int(y_center-height/2)
        y_max = int(y_center+height/2)
        
        crop = img[y_min:y_max, x_min:x_max]
        
        crop_resize = cv2.resize(src = crop, dsize=self.crop_size, interpolation=cv2.INTER_CUBIC)

        return np.array(crop_resize)
    
    def makelabel(self):
        for ind,labelpath in enumerate(self.labels_paths):

            label = self.getlabel(labelpath)
            
            for obj in label:
                self.obj_list.append(obj)
                self.ori_img_list.append(self.images_paths[ind])
        
    def getlabel(self, labelpath):
        clas_list=[]
        box2d_list=[]
        obj_features=[]
        with open(labelpath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            
        for curr_line in lines:
            read_line = curr_line.split(' ')
            clas=int(read_line[0])
            x_center=float(read_line[1])
            y_center=float(read_line[2])
            width=float(read_line[3])
            height=float(read_line[4])
            
            clas_list.append(clas)
            box2d_list.append([x_center,y_center,width,height])
            obj_features.append((clas, [x_center,y_center,width,height]))
            
        return obj_features



