from lib.MOCS_Dataset import MOCSDataset
from lib.utils import parse_yaml
import numpy as np
import cv2

############################################################################################################################################################
# Note: This code is currently not available to be run due to requiring the MOCS dataset which is too big to be uploaded in the GitHub repository
# Code for reformatting the MOCS dataset structure to fit the requirements of YOLO
############################################################################################################################################################


def main(config):
    config = parse_yaml('config.yaml')
    mocsy = MOCSDataset(config, split='test')

    # Usage:
    # mocsy.display_multiple_yolo_segmentations([0, 1, 2, 3, 4, 5])  # Display first 5 images

    mocsy.prep_yolo_seg()

if __name__=='__main__':
    config = parse_yaml('config.yaml')
    main(config)