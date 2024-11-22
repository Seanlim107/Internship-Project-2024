import cv2
import numpy as np
from enum import Enum
import itertools

from lib_3D.File import *
from lib_3D.Math import *

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, box_2d, clip, calib_file=None):
    # if calib_file is not None:
        # cam_to_img = get_calibration_cam_to_image(calib_file)
        # R0_rect = get_R0(calib_file)
        # Tr_velo_to_cam = get_tr_to_velo(calib_file)
        
    minrange=np.array([box_2d[0][0],box_2d[0][1]])
    maxrange=np.array([box_2d[1][0],box_2d[1][1]])

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)
    
    clipped_point = point
    # if clip:
    #     clipped_point = np.clip(point, minrange, maxrange)
    # else:
    #     clipped_point = point
    
    return clipped_point



# take in 3d points and plot them on image as red circles
# def plot_3d_pts(img, pts, center, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None, clip):
#     if calib_file is not None:
#         cam_to_img = get_calibration_cam_to_image(calib_file)

#     for pt in pts:
#         if relative:
#             pt = [i + center[j] for j,i in enumerate(pt)] # more pythonic

#         point = project_3d_pt(pt, cam_to_img, clip)

#         color = cv_colors.RED.value

#         if constraint_idx is not None:
#             color = constraint_to_color(constraint_idx)

#         cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)



def plot_3d_box(img, cam_to_img, ry, dimension, center, box_2d, clip, mode, rz):
    # print(mode)
    colors = [cv_colors.RED.value, cv_colors.ORANGE.value, cv_colors.YELLOW.value, cv_colors.GREEN.value]
    colors_mode = colors[mode]
    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry, roll=rz)

    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img, box_2d, clip)
        box_3d.append(point)
        
    box_3d_rescaled = refine_corners(box_2d=box_2d, box_3d=box_3d)
    box_3d = box_3d_rescaled
    #TODO put into loop
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), colors_mode, 2)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), colors_mode, 2)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), colors_mode, 2)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), colors_mode, 2)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), colors_mode, 2)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), colors_mode, 2)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), colors_mode, 2)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), colors_mode, 2)

    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), colors_mode, 2)

    front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]

    cv2.line(img, front_mark[0], front_mark[3], colors_mode, 2)
    cv2.line(img, front_mark[1], front_mark[2], colors_mode, 2)
    
    return box_3d

def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None, beta=0, clip=False):

    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray, beta)
    
    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img, box_2d)

    box_3d = plot_3d_box(img, cam_to_img, orient, dimensions, location, box_2d, clip, mode=-1, rz=beta) # 3d boxes

    return location, box_3d

def refine_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, location, mode=0, clip=False, beta=0):
    orient = alpha + theta_ray
    box_3d = plot_3d_box(img, cam_to_img, orient, dimensions, location, box_2d, clip, mode, beta) # 3d boxes
    return box_3d

def refine_corners(box_2d, box_3d):
    
    cuboid_2D = np.array(box_3d, dtype=np.float32)
    (xmin, ymin), (xmax, ymax) = box_2d
    rectangle_2D = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmin, ymax],
        [xmax, ymax]
    ], dtype=np.float32)
    # Step 3: Compute the centroids of the cuboid and the rectangle
    cuboid_centroid = np.mean(cuboid_2D, axis=0)
    rectangle_centroid = np.mean(rectangle_2D, axis=0)

    # Step 4: Translate the cuboid to align centroids at the origin
    cuboid_translated = cuboid_2D - cuboid_centroid

    # Step 5: Calculate the dimensions of the cuboid and rectangle
    cuboid_width = np.max(cuboid_translated[:, 0]) - np.min(cuboid_translated[:, 0])
    cuboid_height = np.max(cuboid_translated[:, 1]) - np.min(cuboid_translated[:, 1])

    rectangle_width = np.max(rectangle_2D[:, 0]) - np.min(rectangle_2D[:, 0])
    rectangle_height = np.max(rectangle_2D[:, 1]) - np.min(rectangle_2D[:, 1])

    # Step 6: Calculate scaling factors
    scale_x = rectangle_width / cuboid_width
    scale_y = rectangle_height / cuboid_height
    scale = min(scale_x, scale_y)  # Use the smallest scale to maintain aspect ratio

    # Step 7: Scale the cuboid
    cuboid_scaled = cuboid_translated * scale

    # Step 8: Translate the scaled cuboid back to the rectangle's centroid
    cuboid_final = cuboid_scaled + rectangle_centroid

    return cuboid_final.astype(np.int16)

# def refine_corners(box_2d, box_3d):
#     pt1, pt2 = box_2d
    
#     box_3d_np = np.array(box_3d)

#     box_3d_center = np.mean(box_3d_np, axis=0)
#     box_3d_translated = align_centers(box_3d=box_3d_np, box_2d=box_2d)
#     box_3d_translated_center = np.mean(box_3d_translated, axis=0)
    
#     box_3d_mins = np.min(box_3d_translated, axis=0)
#     box_3d_maxs = np.max(box_3d_translated, axis=0)
    
#     # box_3d_mins = np.min(box_3d_np, axis=0)
#     # box_3d_maxs = np.max(box_3d_np, axis=0)
    
    
#     box_2d_center = np.array([pt2[0]+pt2[0], pt2[1]+pt1[1]])/2
    
#     box_3d_width = box_3d_maxs[0] - box_3d_mins[0]
#     box_3d_height = box_3d_maxs[1] - box_3d_mins[1]
    
#     box_2d_width = pt2[0]-pt1[0]
#     box_2d_height = pt2[1]-pt1[1]
#     # scale = min(box_2d_width/box_3d_width, box_2d_height/box_3d_height)

    
#     scale = min(box_2d_width/box_3d_width, box_2d_height/box_3d_height)
#     # box_3d_rescaled = box_2d_center  + (box_3d_np - box_2d_center) * scale
    
#     box_3d_rescaled = box_3d_translated_center + (box_3d_translated - box_3d_translated_center) * scale
    
    
#     # minrange=np.array([pt1[0],pt1[1]])
#     # maxrange=np.array([pt2[0],pt2[1]])
    
#     # box_3d_clipped = np.clip(box_3d_rescaled, minrange, maxrange)
    
#     # box_3d_clipped = box_3d_clipped.astype(np.int16)
#     box_3d_rescaled = box_3d_rescaled.astype(np.int16)
    
#     return box_3d_rescaled

def align_centers(box_3d, box_2d):
    box_3d_center = np.mean(box_3d, axis=0)
    pt1, pt2 = box_2d
    box_2d_center = np.array([pt2[0]+pt2[0], pt2[1]+pt1[1]])/2
    box_3d_np = np.array(box_3d)
    
    box_3d_new = box_3d_np + box_3d_center - box_2d_center 
    
    return box_3d_new