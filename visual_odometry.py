# https://cmsc733.github.io/2019/proj/p3/
# https://www.youtube.com/watch?v=Opy8xMGCDrE
# https://www.youtube.com/watch?v=olmAyg_mrdI&list=PLZgpos4wVnCYhf5jsl2HcsCl_Pql6Kigk&index=2
# https://github.com/BraulioV/Computer-Vision/blob/master/3.Camera-estimation%2Cepipolar-geometry/camera_calibration.py
# http://cseweb.ucsd.edu/classes/sp19/cse152-a/hw2/HW2.pdf
# https://engineering.purdue.edu/kak/computervision/ECE661_Fall2012/solution/hw8_s2.pdf
# https://en.wikipedia.org/wiki/Eight-point_algorithm
# http://slazebni.cs.illinois.edu/spring19/lec16_epipolar.pdf
# https://docs.opencv.org/master/df/d74/classcv_1_1FastFeatureDetector.html
# https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://medium.com/data-breach/introduction-to-feature-detection-and-matching-65e27179885d
# https://answers.opencv.org/question/61764/problem-extracting-descriptors-of-fast-opencv3/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://www.programcreek.com/python/example/89444/cv2.drawMatches
# https://stackoverflow.com/questions/28351384/dmatch-class-opencv

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
import random
# # ------Data Preparation------
original_images = []
gray_images = []
color_images = []

# in_read = input("Enter 1 to read data, else enter 0 to continue: ")
images_to_read = 50


def read_images():
    print("Reading images...")
    count = 0
    for file in sorted(glob.glob("stereo/centre/*.png")):
        if count <= images_to_read: # Total 3873 images hangs my computer.
            frame = cv.imread(file)
            original_images.append(frame)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)        
            gray_images.append(gray)

            color_img = cv.cvtColor(gray, cv.COLOR_BayerGR2BGR)

            color_images.append(color_img)
        count += 1
            

    print("{0} out of {1} images have been loaded! \n".format(len(color_images) - 1, count))

    fx , fy , cx , cy , G_camera_image, LUT = rcm.ReadCameraModel('./model')
    # print("Camera parameters- \nfx: {0}, fy: {1} \ncx: {2}, cy: {3} \nLUT: \n {4} \nG_image: \n {5} \n".format(fx, fy, cx, cy, LUT, G_camera_image))
    camera_params_f_c = []
    camera_params_f_c.append(fx)
    camera_params_f_c.append(fy)
    camera_params_f_c.append(cx)
    camera_params_f_c.append(cy)

    # # Required for future
    # np.save('camera_params.npy', camera_params_f_c)
    # np.save('g_camera_image.npy', G_camera_image)
    # np.save('LUT.npy', LUT)
    return original_images, gray_images, color_images, LUT

def undistort_images(features1, features2, LUT):
    print("Undistorting images...")
    
    # undistorted_images = []
    # for individual_image in color_images:
    undistorted_feature_1 = ui.UndistortImage(features1, LUT)
    undistorted_feature_2 = ui.UndistortImage(features2, LUT)

    return undistorted_feature_1, undistorted_feature_2

# # ------Main Code------
# Load Camera parameters.
def load_params():
    cam_params = np.load('camera_params.npy')
    g_image = np.load('g_camera_image.npy')
    lut = np.load('LUT.npy')
    print(cam_params, g_image, lut)

def fast_feature_tracking(undistorted_img_1, undistorted_img_2, feature_img_1, feature_img_2):
    print("Tracking features...\n")

    img = undistorted_img_1
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #np.float32(gray_images[0])

    img1 = undistorted_img_2
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) #np.float32(gray_images[0])

    fast = cv.FastFeatureDetector_create(threshold=45, nonmaxSuppression=False) #threshold=35, nonmaxSuppression=False)
    keypoints = fast.detect(gray, None)
    keypoints_second = fast.detect(gray1, None)

    with_features = cv.drawKeypoints(feature_img_1, keypoints, None, color=(0,0,255))
    with_features_second = cv.drawKeypoints(feature_img_2, keypoints_second, None, color=(0,255,0))

    # Print the number of keypoints detected in the training image
    print("Number of Keypoints Detected In The First Image: ", len(keypoints))
    print("Number of Keypoints Detected In The Second Image: ", len(keypoints_second))

    # To create descriptors
    print("Creating Descriptors...")
    br = cv.BRISK_create()
    keypoints, descriptors = br.compute(feature_img_1, keypoints)
    keypoints_second, descriptors_second = br.compute(feature_img_2, keypoints_second)
    
    # BF Matcher
    # # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # # Match descriptors.
    matches = bf.match(descriptors, descriptors_second) #query, train
    # # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # # Draw first 10 matches.
    # print(len(matches), matches[10].imgIdx, matches[10].trainIdx, matches[10].queryIdx)
    corr_points_1, corr_points_2 = [], []
    for i in matches:
        corr_points_1.append(keypoints[i.queryIdx].pt)
        corr_points_2.append(keypoints_second[i.trainIdx].pt)

    # img3 = cv.drawMatches(color_images[34],keypoints,color_images[35],keypoints_second,matches[:300],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    corr_pt1, corr_pt2 = [], []
    for i, j in zip(corr_points_1, corr_points_2):
        corr_pt1.append((int(i[0]), int(i[1])))
        corr_pt2.append((int(j[0]), int(j[1])))
    # print(corr_pt1, "\n", corr_pt2)

    # # Check if points are correct
    # pt1, pt2 =corr_pt1[0][0], corr_pt1[0][1]
    # for i in corr_pt1:
    #     feature_img_1 = cv.circle(feature_img_1, (i[0], i[1]), 1, (0,0,255), 2)

    # while(1):
    #     cv.imshow('features_in_img0',feature_img_1)
    #     # cv.imshow('feat', with_features)
    #     # cv.imshow('feat1', with_features_second)
    #     key = cv.waitKey(1)
    #     if key == 27:
    #         break
    return corr_pt1, corr_pt2

def fundamental_matrix(p1, p2):
    A = np.zeros((8,9))
    for i in range(0, 8):
        x1 = p1[i][0]
        x2 = p2[i][0]
        y1 = p1[i][1]
        y2 = p2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    
    U,S,V = np.linalg.svd(A, full_matrices = True, compute_uv = True)
    f_matrix = V[-1]
    f_matrix = f_matrix.reshape(3,3)
    U_updated, S_updated, V_updated = np.linalg.svd(f_matrix)
    print(S_updated)
    S_updated_new = np.array([[S_updated[0], 0, 0], [0, S_updated[1], 0], [0, 0, 0]])
#    f_matrix = U_updated.dot(S_updated_new.dot(V_updated))
    f_matrix = np.matmul(U_updated, np.matmul(S_updated_new, V_updated))
    print(f_matrix)
    # return f_matrix

def check_threshold(f, x1, x2):
    val = np.matmul(np.matmul(np.array([x2[0], x2[1], 1]).T, f), np.array([x1[0], x1[1], 1]))
    return val

def outlier_detection(features_img1, features_img2, threshold, iterations):
    it = 0
    final_inliers_1 = []
    final_inliers_2 = []
    while it < iterations:
        temp1 = []
        temp2 = []
        CorrectFeatures1 = []
        CorrectFeatures2 = []

        points_random = []
        eight_random_points = []

        inliers_threshold = 0 #store updated inlier count
        inliers_count = 0 #store 

        it = it + 1
        
        #pickup 8 random points        
        while(len(points_random) <= 8):
            num = random.randint(0, len(features_img1) - 1)
            points_random.append(num)
        
        for p in points_random:
            CorrectFeatures1.append([features_img1[p][0], features_img1[p][1]])
            CorrectFeatures2.append([features_img2[p][0], features_img2[p][1]])
        
        EstFundamentalMatrix = fundamental_matrix(CorrectFeatures1, CorrectFeatures2)
        
        #Checking the fundamental matrix on every point of the features
        for j in range(0, len(features_img1)):
            thresh_new = check_threshold(EstFundamentalMatrix, features_img1[j], features_img2[j])
            if thresh_new < threshold:
                print("Check")
                inliers_count = inliers_count + 1
                temp1.append(features_img1[j])
                temp2.append(features_img2[j])
        
        #update if better match found
        if inliers_count > inliers_threshold:
            print("Update")
            inliers_threshold = inliers_count 
            final_inliers_1 = temp1
            final_inliers_2 = temp2
            FinFundamentalMatrix = EstFundamentalMatrix
        
    return final_inliers_1, final_inliers_2, FinFundamentalMatrix

orig_img, gray_img, color_img, LUT_matrix = read_images()
feature_img_1, feature_img_2 = color_img[0], color_img[1]
undistorted_img_1, undistorted_img_2 = undistort_images(feature_img_1, feature_img_2, LUT_matrix)
# load_params()
corres_points_1, corres_points_2 = fast_feature_tracking(undistorted_img_1, undistorted_img_2, feature_img_1, feature_img_2)

iterations = 100
constraint_threshold = 1

fin_inliers_1, fin_inliers_2, fin_funda_mat = outlier_detection(corres_points_1, corres_points_2, constraint_threshold, iterations)
print(fin_inliers_1, fin_inliers_2)
cv.destroyAllWindows()