from torch.utils import data
from torchvision import transforms
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import shutil
import torch
import json
from tqdm import tqdm

class MOCSDataset(data.Dataset):
    def __init__(self, config, crops=True, split='train'):
        self.filedir = os.path.dirname(os.path.dirname(__file__))
        
        self.splits = ['train', 'test', 'val']
        self.filetypes = ['train', 'test', 'valid']
        self.imglabel = ['images', 'labels']
        config_yoloprep = config['Dataset']['MOCS']
        self.movingdir = config_yoloprep['moving_dir']
        self.mainpath = config_yoloprep['main_dir']
        self.resize_size = tuple([config['Dataset']['MOCS']['resize_size'], config['Dataset']['MOCS']['resize_size']])
        self.crop_size = tuple([config['General']['crop_size'], config['General']['crop_size']])
        self.crops = crops
        self.lookup = config['Classes']
        
        # tally for corrupt/good files
        self.good = 0
        self.corrupt = 0
        self.split = split
        instances_dir_name_image =os.path.join(f'instances_{self.split}',f'instances_{self.split}')
        instances_dir_name_label = f'instances_{self.split}.json'
        
        self.input_dir_image = os.path.join(self.filedir, self.mainpath, instances_dir_name_image)
        self.input_labels_dir = os.path.join(self.filedir, self.mainpath, instances_dir_name_label)
        
        self.images_dir = os.path.join(self.filedir, self.movingdir, 'images', self.split)
        self.labels_dir = os.path.join(self.filedir, self.movingdir, 'labels', self.split)
        
        if os.path.exists(self.images_dir):
            self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        else:
            self.image_files = []
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_size),
            
        ])
        
        self.dict = {
            0: "worker", 1: "suspended load", 2: "static crane", 3: "crane",
            4: "roller", 5: "bulldozer", 6: "excavator", 7: "truck",
            8: "loader", 9: "pump truck", 10: "concrete mixer", 11: "pile driving",
            12: "forklift"
        }

    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + '.txt')

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        

        # Load labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    polygon = [float(x) for x in data[1:]]

                    labels.append({'class_id': class_id, 'polygon': polygon})

        

        # Apply transformations
        if self.transform:
            # image = self.transform(image)
            transform_image = self.transform(image)
            
        # else:
        
        # Create target dictionary
        target = {
            'labels': torch.tensor([label['class_id'] for label in labels], dtype=torch.int64),
            'polygons': [label['polygon'] for label in labels],
            # 'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor(image.shape[1:]),
            "imgfilename": img_path, 
            "labelfilename": label_path
        }
        
        return image, target, transform_image
        # return image, target

    def prep_yolo_seg(self):
        self.makedirs()
        # Load JSON data (using the same JSON data as in your original message)
        curr_lb = self.imglabel[1]
        curr_im = self.imglabel[0]
        for iter, curr_split in enumerate(self.splits):
            curr_type = self.filetypes[iter]
            instances_dir_name_label = f'instances_{curr_split}.json'
            dir_name_image = os.path.join(f"instances_{curr_split}", f"instances_{curr_split}")
            input_dir_image = os.path.join(self.filedir, self.mainpath, dir_name_image)
            output_dir_image = os.path.join(self.filedir, self.movingdir, curr_im, curr_type)
            local_labels_dir = os.path.join(self.filedir, self.mainpath, instances_dir_name_label)
            output_dir_label = os.path.join(self.filedir, self.movingdir, curr_lb, curr_type)
            # print(local_labels_dir)
            if(curr_split == 'test'):
                dir_name_image = os.path.join(f"instances_{curr_split}", f"instances_{curr_split}")
                output_dir_image = os.path.join(self.filedir, self.movingdir, curr_im, curr_type)
                input_dir_image = os.path.join(self.filedir, self.mainpath, dir_name_image)
                for _,_,listimgfilename in os.walk(input_dir_image):
                    for imagefilename in listimgfilename:
                        self.copy_images(imagefilename, input_dir_image, output_dir_image)
                continue
            
            with open(local_labels_dir) as json_file:
                json_data = json.load(json_file)


            # Convert JSON to YOLO segmentation format
            self.prepare_label_seg(json_data, output_dir_label, input_dir_image, output_dir_image)
        
        
        
    def makedirs(self):
        for type in self.filetypes:
            for imlb in self.imglabel:
                os.makedirs(os.path.join(self.filedir, self.movingdir, imlb, type), exist_ok=True)
        print('Dirs Created')

    def normalize_coordinates(self, coords, img_width, img_height):
        return [
            coords[i] / img_width if i % 2 == 0 else coords[i] / img_height for i in range(len(coords))
        ]
        
    # def get_label(self, json_data, output_dir)

    def prepare_label_seg(self, json_data, output_dir_label, input_dir_image, output_dir_image):
        

        categories = {cat['id']: idx for idx, cat in enumerate(json_data['categories'])}

        for image in tqdm(json_data['images']):
            corrupt_flag = 0
            img_id = image['id']
            # true_img_id = int(os.path.splitext(image['file_name'])[0])
            
            img_width = image['width']
            img_height = image['height']
            img_filename = image['file_name']
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            output_path = os.path.join(output_dir_label, label_filename)

            with open(output_path, 'w') as f:
                for ann in json_data['annotations']:
                    ann_img_id = ann['image_id']

                    if ann_img_id == img_id: 
                        if corrupt_flag == 1:
                            continue
                        category_idx = categories[ann['category_id']]
                        segmentation = ann['segmentation'][0]  # Assuming single polygon per annotation
                        normalized_seg = self.normalize_coordinates(segmentation, img_width, img_height)
                        
                        
                        
                        if all(coord <= 1 and coord >= 0 for coord in normalized_seg):
                            self.create_yolo_labels(output_path, category_idx, normalized_seg, f)
                        else:
                            corrupt_flag = 1
                        
                    elif ann_img_id > img_id:
                        if corrupt_flag == 0:
                            self.good += 1
                            self.copy_images(img_filename, input_dir_image, output_dir_image)
                        else:
                            f.close()
                            if os.path.isfile(output_path):
                                os.remove(output_path)
                            self.corrupt += 1
                        break
                            
                        

        print(f"Conversion completed. YOLO segmentation format files are saved in {output_dir_label}. Corrupt files: {self.corrupt}, Good files: {self.good}")


    def copy_images(self, img_filename, input_dir_image, output_dir_image):
        source_file = os.path.join(input_dir_image, img_filename)
        dest_file = os.path.join(output_dir_image, img_filename)
        
        shutil.copy(source_file, dest_file)
        
    def display_multiple_yolo_segmentations(self, indices):
        num_images = len(indices)
        rows = int(np.ceil(num_images / 3))
        cols = min(num_images, 3)
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for ax, idx in zip(axes, indices):
            image, target = self[idx]
            image_np = image.permute(1, 2, 0).numpy()
            
            ax.imshow(image_np)
            print(target['labelfilename'], target['imgfilename'])
            
            polygons = []
            for polygon in target['polygons']:
                pixel_coords = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] * image_np.shape[1]
                    y = polygon[i+1] * image_np.shape[0]
                    pixel_coords.append((x, y))
                polygons.append(patches.Polygon(pixel_coords, closed=True, fill=False))
            
            p = PatchCollection(polygons, facecolors='none', edgecolors='r', linewidths=2)
            ax.add_collection(p)
            
            for polygon, label in zip(target['polygons'], target['labels']):
                x = polygon[0] * image_np.shape[1]
                y = polygon[1] * image_np.shape[0]
                class_name = self.dict.get(label.item(), f"Class {label.item()}")
                ax.text(x, y, class_name, color='white', fontsize=8, 
                        bbox=dict(facecolor='red', alpha=0.5))
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage:
# dataset.display_multiple_yolo_segmentations([0, 1, 2, 3, 4])  # Display first 5 images

    def create_yolo_labels(self, output_path, category_idx, normalized_seg, f):
        
        # Write class index and normalized coordinates
        f.write(f"{category_idx} " + " ".join([f"{coord:.6f}" for coord in normalized_seg]) + "\n")


# # Usage example:
# if __name__ == "__main__":
#     config = {
#         'Dataset': {'MOCS': {'moving_dir': 'path/to/moving/dir', 'main_dir': 'path/to/main/dir'}},
#         'General': {'resize_size': 640, 'crop_size': 512},
#         'Classes': {}  # Add your class lookup here if needed
#     }
#     dataset = MOCSDataset(config, split='train')
#     print(f"Dataset size: {len(dataset)}")
    
#     # Get the first item
#     image, target = dataset[0]
#     print(f"Image shape: {image.shape}")
#     print(f"Number of objects: {len(target['labels'])}")
#     print(f"Labels: {target['labels']}")