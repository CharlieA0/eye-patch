import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

# import sys
# sys.executable

# setup
np.set_printoptions(precision=4, suppress=True)
st.title("Online Calibration")

# parse KITTI data with pykitti
import pykitti

basedir = "kitti-data/"
date    = "2011_09_26"
drive   = "0056"

# load synced + rectified data
data = pykitti.raw(basedir, date, drive)

# load unsynced + unrectified data
data_raw = pykitti.raw(basedir, date, drive, dataset="extract")

# grab a stereo pair
cam0, cam1 = data.get_gray(0)
img0 = np.array(cam0) # convert pil image to numpy array
img1 = np.array(cam1)

cam0_raw, cam1_raw = data_raw.get_gray(0)
img0_raw = np.array(cam0_raw)
img1_raw = np.array(cam1_raw)

# # display both pairs of images
"Raw Left Image"
st.image(img0_raw, use_column_width=True)

"Raw Right Image"
st.image(img1_raw, use_column_width=True)
# 
# st.text("Rectified Left Image")
# st.image(img0, use_column_width=True)
# 
# st.text("Rectified Right Image")
# st.image(img1, use_column_width=True)
# 
# # compute disparities
# stereo = cv2.StereoBM_create()
# rect_disparities = stereo.compute(img0, img1)
# raw_disparities  = stereo.compute(img0_raw, img1_raw)
# 
# # display disparity maps
# st.text("Rectified Stereo Disparities")
# plt.imshow(rect_disparities, cmap='viridis')
# st.pyplot()
# 
# st.text("Raw Stereo Disparities")
# plt.imshow(raw_disparities, cmap='viridis')
# st.pyplot()
# 
# rect_density = float((rect_disparities >= 0).sum()) / float(rect_disparities.size)
# raw_density  = float((raw_disparities  >= 0).sum()) / float(raw_disparities.size)
# 
# st.text(f"Rectified Pair Disparity Density: {rect_density:.2%}")
# st.text(f"Raw Pair Disparity Density: {raw_density:.2%}")

# Print calibration fields
# for f in data.calib._fields:
#     f
#     st.text(getattr(data.calib, f))

img_size = data.calib.S_00

# Left image (Which is origin of coordinate system)
D_00 = data.calib.D_00
K_00 = data.calib.K_00
R_00 = data.calib.R_00 # Rotation is identy
T_00 = data.calib.T_00 # Translation is zero

# Right image
D_01 = data.calib.D_01
K_01 = data.calib.K_01
R_01 = data.calib.R_01 # Rotation to left image
T_01 = data.calib.T_01 # Translation to left image

# "Img0"
# st.text(str(D_00))
# st.text(str(K_00))
# st.text(str(R_00))
# st.text(str(T_00))
# 
# "Img1"
# st.text(str(D_01))
# st.text(str(K_01))
# st.text(str(R_01))
# st.text(str(T_01))

undistorted_img0 = cv2.undistort(img0_raw, K_00, D_00)
undistorted_img1 = cv2.undistort(img1_raw, K_01, D_01)

"Undistorted Left Image"
st.image(undistorted_img0, use_column_width=True)

"Undistorted Right Image"
st.image(undistorted_img1, use_column_width=True)

Pr_00 = np.zeros((3,4)) # Left rectify projection matrix
Pr_01 = np.zeros((3,4)) # Right rectify projection matrix
Rr_00 = np.zeros((3,3)) # Left rectify rotation
Rr_01 = np.zeros((3,3)) # Right rectify rotation 

# st.text(
#         (  K_00, 
#                     K_01, 
#                     D_00, 
#                     D_01, 
#                     tuple(img_size), 
#                     R_01, 
#                     T_01, 
#                     Rr_00, 
#                     Rr_01, 
#                     Pr_00, 
#                     Pr_01)  
#         )

# cv2.stereoRectify(  K_00, 
#                     D_00, 
#                     K_01, 
#                     D_01, 
#                     img_size,
#                     R_01, 
#                     T_01, 
#                     Rr_00, 
#                     Rr_01, 
#                     Pr_00, 
#                     Pr_01,  
#                     alpha=0,
#                     )
# 
# st.text((
#     Rr_00,
#     Rr_01,
#     Pr_00,
#     Pr_01,
# ))
# 
# mapx_00, mapy_00 = cv2.initUndistortRectifyMap(K_00, D_00, Rr_00, K_00, img_size, cv2.CV_32FC1)
# img0_rect = cv2.remap(img0_raw, mapx_00, mapy_00, cv2.INTER_LINEAR)
# 
# "Undistorted, Rectified Left Image"
# st.image(img0_rect, use_column_width=True)
# 
# "Kitti Rectified Left Image"
# st.image(img0, use_column_width=True)
# 
# mapx_01, mapy_01 = cv2.initUndistortRectifyMap(K_01, D_01, Rr_01, K_01, img_size, cv2.CV_32FC1)
# img1_rect = cv2.remap(img1_raw, mapx_01, mapy_01, cv2.INTER_LINEAR)
# 
# 'Undistorted, Rectified Right Image'
# st.image(img1_rect, use_column_width=True)
# 
# 'Kitti Rectified Right Image'
# st.image(img1, use_column_width=True)
# 
# # compute disparities
# stereo = cv2.StereoBM_create()
# kitti_disparities = stereo.compute(img0, img1)
# rect_disparities  = stereo.compute(img0_rect, img1_rect)
# 
# # display disparity maps
# "Kitti Rectified Stereo Disparities"
# plt.imshow(kitti_disparities, cmap='viridis')
# st.pyplot()
# 
# "Rectified, Undistorted Stereo Disparities"
# plt.imshow(rect_disparities, cmap='viridis')
# st.pyplot()
# 
# kitti_density = float((kitti_disparities >= 0).sum()) / float(kitti_disparities.size)
# rect_density  = float((rect_disparities  >= 0).sum()) / float(rect_disparities.size)
# 
# f"Rectified Pair Disparity Density: {kitti_density:.2%}"
# f"Raw Pair Disparity Density: {rect_density:.2%}"
