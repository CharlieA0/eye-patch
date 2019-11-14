import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
# 
# st.text("Raw Right Image")
# st.image(img1_raw, use_column_width=True)
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

D_00 = data.calib.D_00
K_00 = data.calib.K_00

undistorted_img0 = cv2.undistort(img0_raw, K_00, D_00)

"Undistorted Left Image"
st.image(undistorted_img0, use_column_width=True)

