# Load a pre-trained YOLOv10n model
# from ultralytics import YOLOv10
from lib.MOCS_Dataset import MOCSDataset
from lib.Dataset import ConstructionDataset
from lib.Camera import Camera
from lib.utils import parse_yaml
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import ultralytics
from ultralytics import YOLO
import torch
from lib.utils import find_xyxy

def iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    boxA: Ground truth bounding box (x1, y1, x2, y2)
    boxB: Predicted bounding box (x1, y1, x2, y2)
    """
    # Coordinates of intersection box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Area of both boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def evaluate_yolo_bbox_performance(ground_truths, predictions, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_scores = []
    
    gt_boxes_xyxy = np.array([find_xyxy(gt_box) for gt_box in ground_truths])
    predictions = predictions.unsqueeze(-1).cpu().numpy()
    matched_gt_boxes = set()

    for pred_box in predictions:
        best_iou = 0
        best_gt_box = None
        
        for gt_box in gt_boxes_xyxy:
            iou_score = iou(gt_box, pred_box)
            if iou_score > best_iou:
                best_iou = iou_score
                best_gt_box = tuple(gt_box.reshape(4))
        
        # If best IoU is above threshold, it's a true positive
        if best_iou >= iou_threshold:
            
            if best_gt_box not in matched_gt_boxes:
                true_positives += 1
                matched_gt_boxes.add(best_gt_box)
                iou_scores.append(best_iou)
            else:
                false_positives += 1  # Double match for the same ground truth box
        else:
            false_positives += 1  # Predicted box did not match sufficiently

    # Count false negatives (unmatched ground truth boxes)
    false_negatives += len(gt_boxes_xyxy) - len(matched_gt_boxes)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    
    return precision, recall, avg_iou, true_positives, false_positives, false_negatives
    
    
def main(config):
    #Initialization
    filedir = os.path.dirname(__file__)
    weightpath = os.path.join(filedir, 'runs/segment/train2/weights/best.pt')
    model = YOLO(weightpath)
    
    # mocsy = MOCSDataset(config, split='test')

    # for image, target, image_tensor in mocsy:
    #     # print(image_tensor)
    #     predictions = model(image_tensor, imgsz=resize_size)
    #     # print(predictions)
    #     annotatedImage = predictions[0].plot()
    #     annotatedImageRGB = cv2.cvtColor(annotatedImage, cv2.COLOR_BGR2RGB)
    #     # ax.imshow(annotatedImageRGB)
    #     plt.imshow(annotatedImageRGB)
    #     plt.show()
        
    consDataset = ConstructionDataset(config, crops=False)
    
    params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 6}
    torch.manual_seed(config['General']['seed'])
    generator = data.DataLoader(consDataset, **params)
    
    total_precision, total_recall, total_avg_iou, total_true_positives, total_false_positives, total_false_negatives = 0,0,0,0,0,0
        
    # Iterate through images and labels and calculate metrics (also show images and segmentation if need be)
    for (indexed_img_tensor, indexed_ori_img, indexed_label, indexed_pair) in generator:

        indexed_img_path, indexed_label_path = indexed_pair
        with torch.no_grad():
            results = model(indexed_img_path)
        
        pred_box = results[0].boxes.xyxy
        
        if(len(pred_box)==0 or len(indexed_label)==0):
            continue
        
        label_class=[]
        label_box=[]
        for local_class, local_box in indexed_label:
            label_class.append(local_class)
            label_box.append(local_box)
        label_box = np.array(label_box)
        wh = np.array([[indexed_ori_img.shape[2], indexed_ori_img.shape[1],indexed_ori_img.shape[2], indexed_ori_img.shape[1]]]).transpose()
        label_box = label_box*wh
        precision, recall, avg_iou, true_positives, false_positives, false_negatives = evaluate_yolo_bbox_performance(ground_truths=label_box,predictions=pred_box)
        
        total_precision+=precision 
        total_recall+=recall 
        total_avg_iou+=avg_iou 
        total_true_positives+=true_positives 
        total_false_positives+=false_positives
        total_false_negatives+=false_negatives
        
        # Plot original Image with Yolo Detection    
        annotatedImage = results[0].plot()
        annotatedImageRGB = cv2.cvtColor(annotatedImage, cv2.COLOR_BGR2RGB)
        plt.imshow(annotatedImageRGB)
        plt.show()
        plt.pause(1)
        plt.clf()
        
    total_len = len(generator)
    avg_precision = total_precision/total_len
    avg_recall = total_recall/total_len
    avg_avg_iou = total_avg_iou/total_len
    avg_true_positives = total_true_positives/total_len
    avg_false_positives = total_false_positives/total_len
    avg_false_negatives = total_false_negatives/total_len
    
    print(f"Avg Precision: {avg_precision}, Avg Recall: {avg_recall}, Avg IOU: {avg_avg_iou}, avg_TP: {avg_true_positives}, avg_FP: {avg_false_positives}, avg_FN: {avg_false_negatives}, total_TP: {total_true_positives}, total_FP: {total_false_positives}, total_FN: {total_false_negatives}")
           
if __name__=='__main__':
    config=parse_yaml('config.yaml')
    
    main(config)


