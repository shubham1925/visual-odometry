import cv2 as cv
import numpy as np
import math
import random

iterations = 100

def FundamentalMatrix(p1, p2):
    A = np.zeros((8,9))
    for i in range(0, 8):
        x1 = p1[i][0]
        x2 = p2[i][0]
        y1 = p1[i][1]
        y2 = p2[i][2]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    
    U,S,V = np.linalg.svd(A, full_matrices = True, compute_uv = True)
    f_matrix = V[-1]
    f_matrix = f_matrix.reshape(3,3)
    U_updated, S_updated, V_updated = np.linalg.svd(f_matrix)
    S_updated_new = np.array([S_updated[0], 0, 0], [0, S_updated[1], 0], [0, 0, 0])
#    f_matrix = U_updated.dot(S_updated_new.dot(V_updated))
    f_matrix = np.matmul(U_updated, np.matmul(S_updated_new, V_updated))
    return f_matrix

def EssentialMatrix(k, f):
    e = np.matmul(k.T, (np.matmul(f, k)))
    u,s,v = np.linalg.svd(e, full_matrices = True, compute_uv = True)
    s_updated = np.array([s[0], 0, 0], [0, s[1], 0], [0, 0, 0])
    e_final = np.matmul(u, np.matmul(s_updated, v))
    return e_final    

def CheckThreshold(f, x1, x2):
    val = np.matmul(np.matmul(np.array([x2[0], x2[1], 1]).T, f), np.array([x1[0], x1[1], 1]))
    return val
        
def CameraPoses(E):
    W = np.array(([0,-1,0], [1, 0, 0], [0, 0, 1]))
    U,D,V = np.linalg.svd(E, full_matrices = True, compute_uv = True)
    C1 = U[:,2]
    C2 = -U[:,2]
    C3 = U[:,2]
    C4 = -U[:,2]
    
    R1 = np.matmul(U, np.matmul(W, V.T))
    R2 = np.matmul(U, np.matmul(W, V.T))
    R3 = np.matmul(U, np.matmul(W.T, V.T))
    R4 = np.matmul(U, np.matmul(W.T, V.T))
    
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
    
    return C1,R1,C2,R2,C3,R3.C4,R4

def OutlierDetect(features_img1, features_img2, threshold, iterations):
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
        
        EstFundamentalMatrix = FundamentalMatrix(CorrectFeatures1, CorrectFeatures2)
        
        #Checking the fundamental matrix on every point of the features
        for j in range(0, len(features_img1)):
            thresh_new = CheckThreshold(EstFundamentalMatrix, features_img1[j], features_img2[j])
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
                
                
            
            
    

    
    
    
    
        