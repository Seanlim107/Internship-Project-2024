from lib.Dataset import ConstructionDataset
from lib.utils import parse_yaml

############################################################################################################################################################
# Note: This code is currently not available to be run due to requiring the ACSD dataset which is not provided as per company restrictions with sensitive data
# Code for reformatting the ACSD dataset to fit YOLO requirements
############################################################################################################################################################


def main(config):
    config_yoloprep = config['Dataset']['Construction']
    mainpath = config_yoloprep['main_dir']
    imgdirname = config_yoloprep['img_dir_name']
    labeldirname = config_yoloprep['lab_dir_name']
    batchdirname = config_yoloprep['Batch_dir_name']
    start_num_batch = config_yoloprep['start_batch_num']
    end_num_batch = config_yoloprep['end_batch_num']
    moving_dir = config_yoloprep['moving_dir']
    img_ext = config_yoloprep['img_ext']
    label_ext = config_yoloprep['label_ext']
    train_size = config_yoloprep['train_size']
    test_size = config_yoloprep['test_size']
    valid_size = config_yoloprep['valid_size']
    init = config_yoloprep['init']
    resize_size = tuple([config['Dataset']['Construction']['resize_size'], config['Dataset']['Construction']['resize_size']])
    seed = config['General']['seed']
    
    consDataset = ConstructionDataset(config, crops=False)
    
    consDataset.createYolo()

if __name__=='__main__':
    config = parse_yaml('config.yaml')
    main(config)