import yaml
import numpy as np
import math
import cv2
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_yaml(config_filename):
    with open(config_filename, 'r') as file:
        configs = yaml.safe_load(file)
        
    return configs

def get_scale(real_height, box_height):
    return real_height/box_height
    
def estimate_distance_2d(img1_xyxy, img2_xyxy, mode=0):
    #__________________________________________________#
    # xyxy in the format of (xmin, ymin, xmax, ymax)
    # Returns minimum distance in pixel length
    #__________________________________________________#
    
    # Check overlapping
    overlap = overlap_check(img1_xyxy,img2_xyxy)
    
    # Minimum distance =0 if bounding boxes overlap
    # if overlap[2]:
    #     return 0
    
    # Find middleground measuring box
    # Referencing # https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares#:~:text=The%20shortest%20distance%20is%20equivalent,by%20that%20of%20both%20squares.  
    xmin1,ymin1,xmax1,ymax1=img1_xyxy[:4]
    width1 =  xmax1-xmin1
    height1 = ymax1-ymin1
    
    xmin2,ymin2,xmax2,ymax2=img2_xyxy[:4]
    width2 = xmax2-xmin2
    height2 = ymax2-ymin2
    
    xmins = (xmin1, xmin2)
    xmaxs = (xmax1, xmax2)
    ymins = (ymin1, ymin2)
    ymaxs = (ymax1, ymax2)
    
    mid_xmin = np.min(xmins)
    mid_xmax = np.max(xmaxs)
    mid_ymin = np.min(ymins)
    mid_ymax = np.max(ymaxs)
    
    
    middle_box = (float(mid_xmin),
                  float(mid_ymin),
                  float(mid_xmax),
                  float(mid_ymax))        
    
    # Calculating horizontal and vertical distance, taking into account possibilities of overlapping on 1 axis
    middle_width = max(0,middle_box[2]-middle_box[0])
    middle_height = max(0,middle_box[3]-middle_box[1])
    
    # Discount vertical or horizontal distances if overlapping vertically or horizontally respectively
    if(overlap[0]):
        between_width = 0
    else:
        between_width = middle_width - width1 - width2
        
    if(overlap[1]):
        between_height = 0
    else:
        between_height = middle_height - height1 - height2
    
    min_distance = np.sqrt(np.power(between_height,2) + np.power(between_width,2))
    
    return float(min_distance)

def clamp_coords(val, min_val, max_val):
    return max(min_val, min(val, max_val))


def estimate_distance_centers_3d(img1_xyz, img2_xyz, mode=0, dim1=None, dim2=None):
    #_______________________________________________________________________#
    # xyxy in the format of (xmin, ymin, xmax, ymax)
    # Returns minimum distance in pixel length
    #_______________________________________________________________________#
    img1_xyz=np.array(img1_xyz[:2])
    img2_xyz=np.array(img2_xyz[:2])
    if dim1 is not None and dim2 is not None:
        dim1 = dim1[:2]/2
        dim2 = dim2[:2]/2
        
        img1_min, img1_max = img1_xyz-dim1, img1_xyz+dim1
        img2_min, img2_max = img2_xyz-dim2, img2_xyz+dim2
        
        # Initialize closest points
        closest_point_img1 = np.zeros(3)
        closest_point_img2 = np.zeros(3)
        
        for i in range(3):
            closest_point_img1[i] = clamp_coords(img2_xyz[i], img1_min[i], img1_max[i])
            closest_point_img2[i] = clamp_coords(img1_xyz[i], img2_min[i], img2_max[i])
            
        distance = np.linalg.norm(closest_point_img1, closest_point_img2)
            
    else:
        closest_point_img1 = img1_xyz
        closest_point_img2 = img2_xyz
    distance = np.linalg.norm(closest_point_img1 - closest_point_img2)

    return distance

# Function for checking if boxes overlap (Used for traditional methods when two bounding boxes are too close to each other in which case distance is 0)
def overlap_check(img1_xyxy, img2_xyxy):
    xmin1, ymin1, xmax1, ymax1 =img1_xyxy[:4]
    
    xmin2, ymin2, xmax2, ymax2 =img2_xyxy[:4]
    
    horizontal_check = [int(xmin1) in range(math.floor(xmin2), math.ceil(xmax2)),
                        int(xmax1) in range(math.floor(xmin2), math.ceil(xmax2)),
                        int(xmin2) in range(math.floor(xmin1), math.ceil(xmax1)),
                        int(xmax2) in range(math.floor(xmin1), math.ceil(xmax1))]
    vertical_check = [int(ymin1) in range(math.floor(ymin2), math.ceil(ymax2)),
                    int(ymax1) in range(math.floor(ymin2), math.ceil(ymax2)),
                    int(ymin2) in range(math.floor(ymin1), math.ceil(ymax1)),
                    int(ymax2) in range(math.floor(ymin1), math.ceil(ymax1))]
    
    return [np.sum(horizontal_check)>1, np.sum(vertical_check)>1, np.sum(horizontal_check)>1 and np.sum(vertical_check)>1]

# Function for drawing lines between two bounding boxes through the center
def draw_lines(img1_xyxy, img2_xyxy, box, dist=None, debug=False, safe_distancing = 50):
    # Calculate the center points of the rectangles
    center1 = find_center(img1_xyxy)
    center2 = find_center(img2_xyxy)
    
    plot_box = (center1[0],center1[1], center2[0], center2[1])
    
    if plot_box is not None:
        xmin= int(plot_box[0])
        ymin= int(plot_box[1])
        xmax= int(plot_box[2])
        ymax= int(plot_box[3])
            
        if(dist < safe_distancing):
            cv2.line(box, (xmin,ymin),(xmax,ymax), color=(0,0,0), thickness=2)
            text_color = (255,0,0)
            
            # Define the text to display
            text = str(round(float(dist), 2)) if dist is not None else ''

        
            mid = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            text_x = mid[0] - text_size[0] // 2
            text_y = mid[1] + text_size[1] // 2
            
            cv2.putText(box, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
        elif(debug):
            cv2.line(box, (xmin,ymin),(xmax,ymax), color=(0,0,0), thickness=2)
            text_color = (255,255,255)
            
            # Define the text to display
            text = str(round(float(dist), 2)) if dist is not None else ''

        
            mid = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            text_x = mid[0] - text_size[0] // 2
            text_y = mid[1] + text_size[1] // 2
            
            cv2.putText(box, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
# Converts formatting of bounding box coordinates from (x,y,w,h) to (x1,y1,x2,y2)
def find_xyxy(box):
    x_center, y_center, w, h = box
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]        
        
# Function for returning center of bounding box assuming (X1,Y1,X2,Y2) formatting
def find_center(box):
    xmin=box[0]
    ymin=box[1]
    xmax=box[2]
    ymax=box[3] 
    
    center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
    return center

# Function for returning angle between two bounding boxes
def get_angle(point1, point2):
    # Returns angle between point 1 (xy) and point 2 (xy) and returns angle in radians
    angle = torch.arctan((point1[1]-point2[1]) / (point1[0]-point2[0]))
    if torch.isnan(angle):
        # Point 1 and Point 2 are the same coordinates, in which case angle shoould be 0
        angle = torch.Tensor([0])
    return angle

def save_checkpoint(epoch, model, model_name, optimizer, loss):
    ckpt = {'epoch': epoch, 
            'model_weights': model.state_dict(), 
            'optimizer_state': optimizer.state_dict(),
            'best_loss': loss}
    torch.save(ckpt, f"checkpoints/{model_name}_ckpt_{str(epoch)}.pth")


def load_checkpoint(model, file_name, optimizer):
    ckpt = torch.load(file_name, map_location=device)
    model_weights = ckpt['model_weights']
    optimizer_state_dict = ckpt['optimizer_state']
    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_state_dict)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    best_loss = ckpt['best_loss']
    epoch = ckpt['epoch']
    print("Model's pretrained weights loaded!")
    
    return best_loss, epoch

# Create bins for angles
def create_bins(num_bins):
    bins = np.array([2*np.pi/num_bins*ind for ind in range(num_bins)])
    return bins

# Use corresponding bins given logit
def get_bin(val, bins, label=None):
    binned_logit = np.digitize(val, bins, right=False)
    
    if(label):
        return bins[binned_logit]
    else:
        return binned_logit
