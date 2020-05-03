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
#https://www.cis.upenn.edu/~cis580/Spring2015/Projects/proj2/proj2.pdf
#https://www-users.cs.umn.edu/~hspark/CSci5980/hw4.pdf

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
import copy
# # ------Data Preparation------
original_images = []
gray_images = []
color_images = []
difference_plot = []

# in_read = input("Enter 1 to read data, else enter 0 to continue: ")
images_to_read = 500


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
    # print(cam_params, g_image, lut)
    return cam_params, g_image, lut

def fast_feature_tracking(undistorted_img_1, undistorted_img_2, feature_img_1, feature_img_2):
    print("Tracking features...\n")

    img = undistorted_img_1
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #np.float32(gray_images[0])

    img1 = undistorted_img_2
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) #np.float32(gray_images[0])

#    fast = cv.FastFeatureDetector_create(threshold=45, nonmaxSuppression=False) #threshold=35, nonmaxSuppression=False)
#    keypoints = fast.detect(gray, None)
#    keypoints_second = fast.detect(gray1, None)

#    with_features = cv.drawKeypoints(feature_img_1, keypoints, None, color=(0,0,255))
#    with_features_second = cv.drawKeypoints(feature_img_2, keypoints_second, None, color=(0,255,0))

#    # Print the number of keypoints detected in the training image
#    print("Number of Keypoints Detected In The First Image: ", len(keypoints))
#    print("Number of Keypoints Detected In The Second Image: ", len(keypoints_second))
#
#    # To create descriptors
#    print("Creating Descriptors...")
#    br = cv.BRISK_create()
#    keypoints, descriptors = br.compute(feature_img_1, keypoints)
#    keypoints_second, descriptors_second = br.compute(feature_img_2, keypoints_second)
    
    orb = cv.ORB_create(nfeatures = 700)
    kp1 = orb.detect(gray, None)
    keypoints, descriptors = orb.compute(feature_img_1, kp1)
    
    kp2 = orb.detect(gray, None)
    keypoints_second, descriptors_second = orb.compute(feature_img_2, kp2)
    
    print("Number of Keypoints Detected In The First Image: ", len(keypoints))
    print("Number of Keypoints Detected In The Second Image: ", len(keypoints_second))
    
    # BF Matcher
    # # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # # Match descriptors.
    matches = bf.match(descriptors, descriptors_second) #query, train
    # # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # # Draw first 10 matches.
    print("Total matches in two images are: {}".format(len(matches)))
    # print(len(matches), matches[10].imgIdx, matches[10].trainIdx, matches[10].queryIdx)
    corr_points_1, corr_points_2 = [], []
    for i in matches:
        corr_points_1.append(keypoints[i.queryIdx].pt)
        corr_points_2.append(keypoints_second[i.trainIdx].pt)

#    img3 = cv.drawMatches(color_images[34],keypoints,color_images[35],keypoints_second,matches[:300],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    plt.imshow("3", img3)
    corr_pt1, corr_pt2 = [], []
    for i, j in zip(corr_points_1, corr_points_2):
        corr_pt1.append((int(i[0]), int(i[1])))
        corr_pt2.append((int(j[0]), int(j[1])))
    # print(corr_pt1, "\n", corr_pt2)

    # # Check if points are correct
    pt1, pt2 =corr_pt1[0][0], corr_pt1[0][1]
    for i in corr_pt1:
        feature_img_1 = cv.circle(feature_img_1, (i[0], i[1]), 1, (0,0,255), 2)
    
    pt1_2, pt2_2 =corr_pt2[0][0], corr_pt2[0][1]
    for i in corr_pt1:
        feature_img_2 = cv.circle(feature_img_2, (i[0], i[1]), 1, (0,0,255), 2)

#    while(1):
#        cv.imshow('features_in_img0',feature_img_1)
#        cv.imshow('features_in_img1',feature_img_2)
#        # cv.imshow('feat', with_features)
#        # cv.imshow('feat1', with_features_second)
#        key = cv.waitKey(1)
#        if key == 27:
#            break
    print("Length of each matched points: {0}, {1}\n".format(len(corr_pt1), len(corr_pt2)))
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
    # print(S_updated)
    S_updated_new = np.array([[S_updated[0], 0, 0], [0, S_updated[1], 0], [0, 0, 0]])
#    f_matrix = U_updated.dot(S_updated_new.dot(V_updated))
    f_matrix = np.matmul(np.matmul(U_updated, S_updated_new), V_updated)
    # print(f_matrix)
    return f_matrix

def check_threshold(f, x1, x2):
    val = np.matmul(np.matmul(np.array([x2[0], x2[1], 1]), f), np.array([x1[0], x1[1], 1]).T)
    return val

def outlier_detection(features_img1, features_img2, threshold, iterations):
    print("Detecting outliers...")
    it = 0
    final_inliers_1 = []
    final_inliers_2 = []
    repeated = []
    
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
            if num not in points_random:# and num not in repeated:
                points_random.append(num)
                repeated.append(num)
        
        for p in points_random:
            CorrectFeatures1.append([features_img1[p][0], features_img1[p][1]])
            CorrectFeatures2.append([features_img2[p][0], features_img2[p][1]])
        
        EstFundamentalMatrix = fundamental_matrix(CorrectFeatures1, CorrectFeatures2)
        
        #Checking the fundamental matrix on every point of the features
        for j in range(0, len(features_img1)):
            thresh_new = check_threshold(EstFundamentalMatrix, features_img1[j], features_img2[j])
#            print("difference: ", np.linalg.norm(thresh_new))
#            difference_plot.append(np.linalg.norm(thresh_new))
            if np.linalg.norm(thresh_new) < threshold:
                # print("Check")
                inliers_count = inliers_count + 1
                temp1.append(features_img1[j])
                temp2.append(features_img2[j])
        
        #update if better match found
        if inliers_count > inliers_threshold:
            # print("Update")
            inliers_threshold = inliers_count 
            final_inliers_1 = temp1
            final_inliers_2 = temp2
            FinFundamentalMatrix = EstFundamentalMatrix
    print("After RANSACwa length of both: {0}, {1} \n".format(len(final_inliers_1), len(final_inliers_2))) 
#    plt.scatter(iterations, difference_plot)    
    return final_inliers_1, final_inliers_2, FinFundamentalMatrix

def essential_matrix(k, f):
    print("Computing Essential Matrix... \n")
    e = np.matmul(np.matmul(k.T, f),k)
    u,s,v = np.linalg.svd(e, full_matrices = True, compute_uv = True)
    s_updated = np.array([[s[0], 0, 0], [0, s[1], 0], [0, 0, 0]])
    e_final = np.matmul(np.matmul(u, s_updated), v)
    return e_final    

def camera_poses(E):
    print("Checking camera poses...")
    W = np.array([[0,-1,0], [1, 0, 0], [0, 0, 1]])
    U,D,V = np.linalg.svd(E, full_matrices = True, compute_uv = True)
    C1 = U[:,2]
    C2 = -U[:,2]
    C3 = U[:,2]
    C4 = -U[:,2]
    
    R1 = np.matmul(np.matmul(U,W), V)
    R2 = np.matmul(np.matmul(U,W), V)
    R3 = np.matmul(np.matmul(U,W.T), V)
    R4 = np.matmul(np.matmul(U,W.T), V)
    
    
    if np.linalg.det(R1) == -1:
        C1 = -C1
        R1 = -R1
    if np.linalg.det(R2) == -1:
        C2 = -C2
        R2 = -R2
    if np.linalg.det(R3) == -1:
        C3 = -C3
        R3 = -R3
    if np.linalg.det(R4) == -1:
        C4 = -C4
        R4 = -R4
    
    return C1,R1,C2,R2,C3,R3,C4,R4

def camera_matrices(C,R):
    P1 = np.identity(4)
    P1 = P1[0:3, :]
    P2 = np.hstack((R, C))
#    P2 = np.matmul(K, P2)
#    P2 = np.vstack(P2, [0,0,0,1])
    return P1, P2

#input p1 and p2 for camera, and matching feature point to get 3d point
def triangulated_point(P1, P2, p_img1, p_img2):
    x1_skew = np.array([[0, -1, p_img1[1]], [1, 0, -p_img1[0]], [-p_img1[1], -p_img1[0], 0]])
    x2_skew = np.array([[0, -1, p_img2[1]], [1, 0, -p_img2[0]], [-p_img2[1], -p_img2[0], 0]])
    
    new_1 = np.matmul(x1_skew, P1[0:3, :])
    new_2 = np.matmul(x2_skew, P2)
    new_final = np.vstack((new_1, new_2))
    U,S,V = np.linalg.svd(new_final)
    X_3D = V[-1]
#    print("orig: ", X_3D)
    #Convert to homogenous
    X_3D = X_3D/X_3D[3]
    X_3D = X_3D.reshape((4,1))
#    print("orig: ", X_3D)
    X_3D = X_3D[0:3].reshape((3,1))
    return X_3D   

def camera_pose_disambiguation(Cset, Rset, Xset):
    print("cam_dis")
    threshold_count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    counts = []
    r3_1 = Rset[0][2, :].reshape((1,3))
    r3_2 = Rset[1][2, :].reshape((1,3))
    r3_3 = Rset[2][2, :].reshape((1,3))
    r3_4 = Rset[3][2, :].reshape((1,3))
#    print(r3_1.shape)
    
    C_1 = Cset[0].reshape((3,1))
    print(C_1.shape)
    C_2 = Cset[1].reshape((3,1))
    C_3 = Cset[2].reshape((3,1))
    C_4 = Cset[3].reshape((3,1))
    
    X_1 = Xset[0]
    print(X_1[0].shape)
    X_2 = Xset[1]
    X_3= Xset[2]
    X_4 = Xset[3]
    
    for i in range(0, len(X_1)):
        diff = np.matmul(r3_1, (X_1[i] - C_1))
        if diff > 0: #np.linalg.norm(diff) > 0:
            count1 = count1 + 1
    for i in range(0, len(X_2)):
#        print("2")
        diff = np.matmul(r3_2, (X_2[i] - C_2))
        if diff > 0: #np.linalg.norm(diff) > 0:
            count2 = count2 + 1
    for i in range(0, len(X_3)):
#        print("3")
        diff = np.matmul(r3_3, (X_3[i] - C_3))
        if diff > 0: #np.linalg.norm(diff) > 0:
            count3 = count3 + 1
    for i in range(0, len(X_4)):
#        print("4")
        diff = np.matmul(r3_4, (X_4[i] - C_4))
        if diff > 0: #np.linalg.norm(diff) > 0:
            count4 = count4 + 1
    counts.append(count1)
    counts.append(count2)
    counts.append(count3)
    counts.append(count4)  
    print("Inside this")
    print(counts)
    print(count1, count2, count3, count4)
    if counts.index(max(counts)) == 0:
        print("1")
        count = []
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        return C1, R1
    if counts.index(max(counts)) == 1:
        print("2")
        count = []
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        return C2, R2
    if counts.index(max(counts)) == 2:
        print("3")
        count = []
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        return C3, R3
    if counts.index(max(counts)) == 3:
        print("4")
        count = []
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        return C4, R4 
    
def final_matrix(R,C):
    temp1 = np.hstack((R,C))
    temp2 = np.array((0,0,0,1))
    final = np.vstack((temp1, temp2))
    return final
    
 
all_3d_1 = []
all_3d_2 = []
all_3d_3 = []
all_3d_4 = []

C_set = []
R_set = []
X_set = []

prev_H = np.eye(4)

orig_img, gray_img, color_img, LUT_matrix = read_images()
all_points = []

for i in range(50, images_to_read-400):
    feature_img_1, feature_img_2 = color_img[i], color_img[i+1]
    undistorted_img_1, undistorted_img_2 = undistort_images(feature_img_1, feature_img_2, LUT_matrix)
    # load_params()
    corres_points_1, corres_points_2 = fast_feature_tracking(undistorted_img_1, undistorted_img_2, feature_img_1, feature_img_2)
    
    iterations = 500
    constraint_threshold = 0.01
    
    inliers_1, inliers_2, final_funda_matrix = outlier_detection(corres_points_1, corres_points_2, constraint_threshold, iterations)
    
    camera_parameters, g_image, lut_load = load_params()
    # print(camera_parameters)
    
    intrinsic_matrix = np.array([[camera_parameters[0], 0, camera_parameters[2]],
                                [0, camera_parameters[1], camera_parameters[3]],
                                [0, 0, 1]])
    
    return_essential_matrix = essential_matrix(intrinsic_matrix, final_funda_matrix)
    # print(return_essential_matrix)
    C1,R1,C2,R2,C3,R3,C4,R4 = camera_poses(return_essential_matrix)
#    print(R1, "\n", R2,"\n", R3,"\n", R4)
    C_set = [C1, C2, C3, C4]
    R_set = [R1, R2, R3, R4]
    
    P1_1, P2_1 = camera_matrices(C1.reshape((3,1)), R1)
    P1_2, P2_2 = camera_matrices(C2.reshape((3,1)), R2)
    P1_3, P2_3 = camera_matrices(C3.reshape((3,1)), R3)
    P1_4, P2_4 = camera_matrices(C4.reshape((3,1)), R4)
#    print("1 size: ", P1_1.shape)
#    print("2 size: ", P2_1.shape)
    
    for i in range(0, len(inliers_1)):
        X_1 = triangulated_point(P1_1, P2_1, inliers_1[i], inliers_2[i])
        all_3d_1.append(X_1)
        X_2 = triangulated_point(P1_2, P2_2, inliers_1[i], inliers_2[i])
        all_3d_2.append(X_2)
        X_3 = triangulated_point(P1_3, P2_3, inliers_1[i], inliers_2[i])
        all_3d_3.append(X_3)
        X_4 = triangulated_point(P1_4, P2_4, inliers_1[i], inliers_2[i])
        all_3d_4.append(X_4)
        
    X_set = [all_3d_1, all_3d_2, all_3d_3, all_3d_4]
    
    C_final, R_final = camera_pose_disambiguation(C_set, R_set, X_set)
    
    final_homogenous_matrix = final_matrix(R_final, C_final.reshape((3,1)))
    
    prev_H = np.matmul(prev_H, final_homogenous_matrix)
    
    point = np.matmul(prev_H, np.array([[0,0,0,1]]).T)
    print(point)
    all_points.append(point)      
    plt.scatter(point[0][0], point[1][0], color = 'b')
    C_set.clear()
    R_set.clear()
    X_set.clear()
    # for i in inliers_1:
    #     feature_img_1 = cv.circle(feature_img_1, (i[0], i[1]), 1, (0,0,255), 2)
    
    # for i in inliers_2:
    #     feature_img_2 = cv.circle(feature_img_2, (i[0], i[1]), 1, (0,0,255), 2)
    
    # while(1):
    #     cv.imshow('features_in_img0',feature_img_1)
    #     cv.imshow('features_in_img1',feature_img_2)
    #     key = cv.waitKey(1)
    #     if key == 27:
    #         break
    
    # keypoint_1 = [cv.KeyPoint(kp[0], kp[1], 1) for kp in inliers_1]
    # keypoint_2 = [cv.KeyPoint(kp[0], kp[1], 1) for kp in inliers_2]
    # print(keypoint_1)
    # br = cv.BRISK_create()
    # keypoints, descriptors = br.compute(feature_img_1, inliers_1)
    # keypoints_second, descriptors_second = br.compute(feature_img_2, inliers_2)
#plt.plot(all_points[0], all_points[1])
plt.show()
cv.destroyAllWindows()