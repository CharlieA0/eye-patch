import streamlit as st
import numpy as np

st.title("Online Calibration")

import pykitti

basedir = "kitti-data/"
date    = "2011_09_26"
drive   = "0056"

data = pykitti.raw(basedir, date, drive)

cam0_img, cam1_img = data.get_gray(0)

st.text("Example Image")
img_np = np.array(cam0_img)
st.image(img_np, use_column_width=True)
