############################################################################################################################################################
# Note: This code is currently not available to be run due to requiring the ACSD dataset which is not provided as per company restrictions with sensitive data
# Code for training the YOLO model on the ACSD dataset
############################################################################################################################################################


# Load a pre-trained YOLOv10n model
from ultralytics import YOLO
from lib.utils import parse_yaml

    # Code for training Yolov10 (scrapped due to complexity)
    # model = YOLOv10()
    # If you want to finetune the model with pretrained weights, you could load the 
    # pretrained weights like below
    # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
    # or
    # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
    # model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')
    # Choose which model you want to use. remember larger the model higher the computational cost
    
def main(configs):
    
    model = YOLO('yolov8n.pt')
    config_yolo = configs['Yolo']
    epochs = config_yolo['epochs']       
    batch = config_yolo['batch'] 
    resize_size = config['Yolo']['resize_size']
    model.train(data='construction_yolo.yaml', epochs=epochs, batch=batch, imgsz=resize_size, patience=300, dropout=0.2)

if __name__=='__main__':
    config = parse_yaml('config.yaml')
    main(config)

