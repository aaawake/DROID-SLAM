import numpy as np
import scipy.misc
import cv2
import os
 
# DF1
path = "/home/pi/工作/predict1/"
npy_list = os.listdir(path)
save_path = "/home/pi/predict1_img/"
if not os.path.exists(save_path):
 os.mkdir(save_path)
 
for i in range(0, len(npy_list)):
 print(i)
 print(npy_list[i])
 npy_full_path = os.path.join(path, npy_list[i])
 img = np.load(npy_full_path) # load进来
 
 save_full_path = os.path.join(save_path, npy_list[i][:-4])
 scipy.misc.imsave(save_full_path, img) # 保存