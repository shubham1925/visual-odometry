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
#https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
#https://www.programcreek.com/python/example/89440/cv2.FlannBasedMatcher

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
import math


images = []
path = "stereo/centre/" 
for image in os.listdir(path): 
    images.append(image) 
    images.sort() 


def check_rotation_matrix(R_current) :
    R_transpose = np.transpose(R_current)
    is_it_identity = np.dot(R_transpose, R_current)
    I = np.identity(3, dtype = R_current.dtype)
    n = np.linalg.norm(I - is_it_identity)
    return n < 1e-6

def convert_rotation_to_euler(R) :
    assert(check_rotation_matrix(R))
    sqrt_sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sqrt_sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sqrt_sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sqrt_sy)
        z = 0
 
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

def undistort_images(features1, features2, LUT):
    print("Undistorting images...")
    undistorted_feature_1 = ui.UndistortImage(features1, LUT)
    undistorted_feature_2 = ui.UndistortImage(features2, LUT)
    return undistorted_feature_1, undistorted_feature_2

def feature_matching(colorimage1, colorimage2):
    features1 = [] 
    features2 = []

    lowe_threshold = 0.75
    undistorted_img_1, undistorted_img_2 = undistort_images(colorimage1, colorimage2, LUT)
    gray1 = cv.cvtColor(undistorted_img_1,cv.COLOR_BGR2GRAY)     
    gray2 = cv.cvtColor(undistorted_img_2,cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures = 3000)
    kp1, des1 = orb.detectAndCompute(gray1,None)
    kp2, des2 = orb.detectAndCompute(gray2,None)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)    

    for i,(m,n) in enumerate(matches):
        if m.distance < lowe_threshold*n.distance:
            features1.append(kp1[m.queryIdx].pt)
            features2.append(kp2[m.trainIdx].pt)
    
    # matches = sorted(matches[1], key = lambda x:x.distance)
    # print("Total matches in two images are: {}".format(len(matches)))
    # img3 = cv.drawMatches(colorimage1, kp1, colorimage2, kp2, matches[:300], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    pt1, pt2 = features1[0][0], features1[0][1]
    for i in features1:
        feature_img_1 = cv.circle(colorimage1, (int(i[0]), int(i[1])), 1, (0,0,255), 2)
    
    pt1_2, pt2_2 =features2[0][0], features2[0][1]
    for i in features2:
        feature_img_2 = cv.circle(colorimage2, (int(i[0]), int(i[1])), 1, (0,0,255), 2)

    img1 = cv.resize(feature_img_1, (0, 0), None, .60, .60)
    img2 = cv.resize(feature_img_2, (0, 0), None, .60, .60)
    
    np_horizontal_concat = np.concatenate((img1, img2), axis = 1)
    cv.imshow("Comparison", np_horizontal_concat)
    # cv.imshow("Matches", img3)
    cv.waitKey(100)
    # key = cv.waitKey(1)
    # if key == 27:
    #     break

    return features1, features2

def fundamental_matrix(p1, p2):
    A = np.empty((8, 9))
    for i in range(0, 8):
        x1 = p1[i][0]
        x2 = p2[i][0]
        y1 = p1[i][1]
        y2 = p2[i][1]
        A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    U,S,V = np.linalg.svd(A, full_matrices=True)
    f_matrix = V[-1].reshape(3,3)    
    U_updated, S_updated, V_updated = np.linalg.svd(f_matrix) 
    S_updated_new = np.array([[S_updated[0], 0, 0], [0, S_updated[1], 0], [0, 0, 0]])
    f_matrix = U_updated @ S_updated_new @ V_updated   
    return f_matrix  

def check_threshold(f, x1, x2): 
    diff = abs(np.squeeze(np.matmul((np.matmul(np.array([x2[0],x2[1],1]),f)),np.array([x1[0],x1[1],1]).T)))
    return diff
    
def essential_matrix(k,f):
    e = np.matmul(np.matmul(k.T, f), k)
    u, s, v = np.linalg.svd(e, full_matrices=True, compute_uv = True)
    s_updated = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    e_final = np.matmul(np.matmul(u, s_updated), v)
    return e_final

def final_matrix(R, t):
    z = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    z = np.vstack((z, a))
    return z


def cam_pose_estimation(essentialMatrix):
    u, s, v = np.linalg.svd(essentialMatrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    c1 = u[:, 2] 
    r1 = u @ w @ v
    c2 = -u[:, 2]
    r2 = u @ w @ v
    c3 = u[:, 2]
    r3 = u @ w.T @ v
    c4 = -u[:, 2]
    r4 = u @ w.T @ v
    
    if np.linalg.det(r1) < 0:
        c1 = -c1 
        r1 = -r1        
    if np.linalg.det(r2) < 0:
        c2 = -c2 
        r2 = -r2    
    if np.linalg.det(r3) < 0:
        c3 = -c3 
        r3 = -r3     
    if np.linalg.det(r4) < 0:
        c4 = -c4 
        r4 = -r4 
    c1 = c1.reshape((3,1))
    c2 = c2.reshape((3,1))
    c3 = c3.reshape((3,1)) 
    c4 = c4.reshape((3,1))
    
    return c1,r1,c2,r2,c3,r3,c4,r4

def outlier_rejection(features_img1, features_img2, threshold, iterations):
    inlier_threshold = 0
    finalFundMatrix = np.zeros((3,3))
    inlier1 = []
    inlier2 = []
    for i in range(0, iterations): 
        temp1 = [] 
        temp2 = []
        CorrectFeatures1 = [] 
        CorrectFeatures2 = []
        
        inlier_count = 0
        points_random = []    
        
        
        while(len(points_random) != 8): 
            num = random.randint(0, len(features1)-1)
            if num not in points_random:
                points_random.append(num)
#        print("len: ",len(points_random))
        for point in points_random: 
            CorrectFeatures1.append([features_img1[point][0], features_img1[point][1]]) 
            CorrectFeatures2.append([features_img2[point][0], features_img2[point][1]])    
        EstFundamentalMatrix = fundamental_matrix(CorrectFeatures1, CorrectFeatures2)
        for number in range(0, len(features_img1)):
            if check_threshold(EstFundamentalMatrix, features1[number], features2[number]) < threshold:
                inlier_count = inlier_count + 1 
                temp1.append(features1[number])
                temp2.append(features2[number])

        if inlier_count > inlier_threshold: 
            inlier_threshold = inlier_count
            finalFundMatrix = EstFundamentalMatrix
            inlier1 = temp1
            inlier2 = temp2
        
    return finalFundMatrix, inlier1, inlier2

def triangulated_point(P1, P2, point1, point2):
    x1_skew = np.array([[0, -1, point1[1]], [1, 0, -point1[0]], [-point1[1], point1[0], 0]]) 
    x2_skew = np.array([[0, -1, point2[1]], [1, 0, -point2[0]], [-point2[1], point2[0], 0]])
    
    new_1 = x1_skew @ P1[0:3, :] 
    new_2 = x2_skew @ P2
    A = np.vstack((new_1, new_2))    
    U, S, V = np.linalg.svd(A)
    X_3D = V[-1]
    X_3D = X_3D/X_3D[3]
    X_3D = X_3D.reshape((4,1))
    X_3D = X_3D[0:3].reshape((3,1))
    
    return X_3D

def camera_pose_disambiguation(R_set, C_set, features1, features2, upper, lower):
    global c_prev, r_prev
    flag = 0
    threshold = 0
    P1 = np.identity(4)
    P1 = P1[0:3, :]
    for r in range(0, len(R_set)):
        angles = convert_rotation_to_euler(R_set[r])
        print(angles[0], angles[2])
        if angles[0] < upper and angles[0] > lower and angles[2] < upper and angles[2] > lower: 
            count = 0 
            P2 = np.hstack((R_set[r], C_set[r]))
            for i in range(0, len(features1)):
                X_3D = triangulated_point(P1, P2, features1[i], features2[i])
                r3 = R_set[r][2,:]
                r3 = r3.reshape((1,3)) 
                C = C_set[r]
                diff = np.squeeze(r3 @ (X_3D - C))
                if diff > 0:
                    count = count + 1 

            if count > threshold:
                flag = 1
                c_prev = C_set[r] 
                r_prev = R_set[r]
                threshold = count
                c_final = C_set[r]
                r_final = R_set[r]

    if flag == 1:
        if c_final[2] > 0:
            c_final = -c_final

    if flag != 1:
        c_final = c_prev
        r_final = r_prev

    return r_final, c_final
    
first_matrix = np.identity(4)
first_point = np.array([[0, 0, 0, 1]]).T 
fx, fy, cx, cy, G_camera_image, LUT = rcm.ReadCameraModel('model/')
K = np.array([[fx , 0 , cx],
              [0 , fy , cy],
              [0 , 0 , 1]]) 

for index in range(6, 100): 
    print("img:",index)
    img1 = cv.imread("stereo/centre/" + str(images[index]), 0) 
    img2 = cv.imread("stereo/centre/" + str(images[index + 1]), 0)

    rgb_img_1 = cv.cvtColor(img1, cv.COLOR_BayerGR2BGR)
    rgb_img_2 = cv.cvtColor(img2, cv.COLOR_BayerGR2BGR) 

    features1, features2 = feature_matching(rgb_img_1, rgb_img_2)
    fin_fund_matrix, inlier1, inlier2 = outlier_rejection(features1, features2, 0.15, 100)
    fin_essential_matrix = essential_matrix(K, fin_fund_matrix)
    c1,r1,c2,r2,c3,r3,c4,r4 = cam_pose_estimation(fin_essential_matrix)
    R_set = [r1,r2,r3,r4]
    C_set = [c1,c2,c3,c4]
    r_final, c_final = camera_pose_disambiguation(R_set, C_set, inlier1, inlier2, 50, -50) 
    first_matrix = first_matrix @ final_matrix(r_final, c_final) 
    p = first_matrix @ first_point 
    plt.scatter(p[0][0], -p[2][0], color='b')
    inlier1.clear()
    inlier2.clear()

plt.show()        
cv.destroyAllWindows()