# https://cmsc733.github.io/2019/proj/p3/
# https://www.youtube.com/watch?v=Opy8xMGCDrE
# https://www.youtube.com/watch?v=olmAyg_mrdI&list=PLZgpos4wVnCYhf5jsl2HcsCl_Pql6Kigk&index=2
# https://github.com/BraulioV/Computer-Vision/blob/master/3.Camera-estimation%2Cepipolar-geometry/camera_calibration.py
# http://cseweb.ucsd.edu/classes/sp19/cse152-a/hw2/HW2.pdf
# https://engineering.purdue.edu/kak/computervision/ECE661_Fall2012/solution/hw8_s2.pdf
# https://en.wikipedia.org/wiki/Eight-point_algorithm
# http://slazebni.cs.illinois.edu/spring19/lec16_epipolar.pdf

import sys, os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2 as cv
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import glob
import ReadCameraModel as rcm
import UndistortImage as ui

# # ------Data Preparation------
original_images = []
gray_images = []
all_images = []
count = 0
# in_read = input("Enter 1 to read data, else enter 0 to continue: ")
images_to_read = 50



print("Reading images...")
for file in glob.glob("stereo/centre/*.png"):
    count += 1
    if count <= images_to_read: # Total 3873 images hangs my computer.
        frame = cv.imread(file)
        original_images.append(frame)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)        
        gray_images.append(gray)

        color_img = cv.cvtColor(gray, cv.COLOR_BayerGR2BGR)

        all_images.append(color_img)
        

print("{0} out of {1} images have been loaded! \n".format(len(all_images), count))

fx , fy , cx , cy , G_camera_image, LUT = rcm.ReadCameraModel('./model')
print("Camera parameters- \nfx: {0}, fy: {1} \ncx: {2}, cy: {3} \nLUT: \n {4} \nG_image: \n {5} \n".format(fx, fy, cx, cy, LUT, G_camera_image))
camera_params_f_c = []
camera_params_f_c.append(fx)
camera_params_f_c.append(fy)
camera_params_f_c.append(cx)
camera_params_f_c.append(cy)

# # Required for future
# np.save('camera_params.npy', camera_params_f_c)
# np.save('g_camera_image.npy', G_camera_image)
# np.save('LUT.npy', LUT)


print("Undistorting images...")
print("{0} images may take a while... Please stay calm.! \n".format(images_to_read))

undistorted_images = []
for individual_image in all_images:
    und_image = ui.UndistortImage(individual_image, LUT)
    undistorted_images.append(und_image)

# # Tried to resize all but UndistortImage does not work for all 3873

# # # Resize while preserving aspect ratio.
# scale_percentage = 40 #percentage of original size
# width = int(all_images[0].shape[1] * scale_percentage / 100)
# height = int(all_images[0].shape[0] * scale_percentage / 100)
# dim = (width, height)

# resized_orig_distorted_img = cv.resize(original_images[34], dim, interpolation=cv.INTER_AREA)
# resized_color_distorted_img = cv.resize(all_images[34], dim, interpolation=cv.INTER_AREA)
# resized_color_undistorted_img = cv.resize(undistorted_images[34], dim, interpolation=cv.INTER_AREA)

# print("One of the 'resized' sample is...")
# while(1):
#     cv.imshow('Original', resized_orig_distorted_img)
#     cv.imshow('Distorted', resized_color_distorted_img)
#     cv.imshow('Undistorted', resized_color_undistorted_img)
#     key = cv.waitKey(1)
#     if key == 27:
#         break

# # ------Main Code------
# Load Camera parameters.

cam_params = np.load('camera_params.npy')
g_image = np.load('g_camera_image.npy')
lut = np.load('LUT.npy')
print(cam_params, g_image, lut)

print("Tracking features...\n")
img = undistorted_images[34]
gray = np.float32(gray_images[34])
corners = cv.goodFeaturesToTrack(gray,15,0.001,50) #maxCorners, quality, minDistance
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(all_images[34], (x,y), 3, [0,0,255], -1)

while(1):
    cv.imshow('features',all_images[34])
    if cv.waitKey(0) & 0xff == 27:
        break

cv.destroyAllWindows()