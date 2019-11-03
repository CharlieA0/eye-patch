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

data = pykitti.raw(basedir, date, drive)

# example stereo matching
cam0_img, cam1_img = data.get_gray(0)

st.text("Left Image")
img0 = np.array(cam0_img)
st.image(img0, use_column_width=True)

st.text("Right Image")
img1 = np.array(cam1_img)
st.image(img1, use_column_width=True)

# compute disparities
stereo = cv2.StereoBM_create()
disparities = stereo.compute(img0, img1)

st.text("Stereo Disparities")
plt.imshow(disparities, cmap='viridis')
st.pyplot()

# st.text(disparities)
# st.image(disparities, use_column_width=True)

# Print calibration fields
# for f in data.calib._fields:
#     st.text(f)
# st.text(data.calib.P_rect_00)

