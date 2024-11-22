import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.utils import find_center

class Camera:
    def __init__(self, use_own, img=None, distortion_coef=None, fx=None, fy=None, cx=None, cy=None, tx=None, ty=None, tz=None):
        if use_own:
            if not [x for x in (fx,fy,cx,cy) if (x is None or x < 0)]:
                self.fx=float(fx)
                self.fy=float(fy)
                self.cx=float(cx)
                self.cy=float(cy)
            else:
                raise Exception('Fx, Fy, Cx and Cy must be Greater than 0!')
        else:
            if img is not None:
                H,W = img.shape[:2]
                
                self.cx=W/2
                self.cy=H/2
                self.fx=min(W,H)
                self.fy=min(W,H)
            else:
                raise Exception('Unexpected Error Occured')
        
        self.K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1]])
        
        self.distortion_coef = np.array(distortion_coef)
    
        
    def remove_fisheye(self, img):
        H,W = img.shape[:2]
        if self.distortion_coef is None:
            return img
        else:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.distortion_coef, (W, H), np.eye(3), balance=1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.distortion_coef, np.eye(3), self.K, (W, H), cv2.CV_32FC1)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            
            
            return undistorted_img

    def find_depth(self, img_xyxy, obj_dim):
        # Finds distance between a detected object and the camera
        xmin,ymin,xmax,ymax = img_xyxy[:4]
        pixel_length = abs(xmax-xmin)
        pixel_height = abs(ymax - ymin)
        real_len, _, real_height = obj_dim
        
        # Find depth relative to heights and widths and average them
        depth_y = self.fy * real_height / pixel_height
        # depth_x = self.fx * real_len / pixel_length
        
        # depth = (depth_x + depth_y)/2
        depth = depth_y
        
        return depth
    
    def get_scale(self,img_xyxy, obj_dim):
        # Finds distance between a detected object and the camera
        xmin,ymin,xmax,ymax = img_xyxy[:4]
        # pixel_length = abs(xmax-xmin)
        pixel_height = abs(ymax - ymin)
        real_len, real_width, real_height = obj_dim
        
        # Find depth relative to heights and widths and average them
        scale = real_height / pixel_height
        # depth_x = self.fx * real_len / pixel_length
        
        
        return scale
    
    def find_real_coords(self, img_xyxy, obj_dim):
        # Takes in a tuple of x y x y coordinates and returns a tuple of xyz xyz coordinates
        depth = self.find_depth(img_xyxy, obj_dim)
        center_x, center_y = find_center(img_xyxy)
        
        X = (center_x - self.cx) * depth / self.fx
        Y = (center_y - self.cy) * depth / self.fy
        
        return (X,Y,depth)
            
        
    