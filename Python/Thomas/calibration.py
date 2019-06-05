#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:59:09 2019

@author: thomaspaita
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#le nombre de colonne et de ligne doit être exactement le nombre
#de coin intérieur de la mire de calibration
c = 9
l = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((l*c,3), np.float32)
objp[:,:2] = np.mgrid[0:c,0:l].T.reshape(-1,2)*27

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
gray=[]

images = glob.glob('data/calib/*.png')

for i,fname in enumerate(images):
    print(i)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (c,l),cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("## Founded ##")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
# =============================================================================
#         img = cv2.drawChessboardCorners(img, (c,l), corners2,ret)
#         plt.figure(i)
#         plt.axis('off')
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# =============================================================================


#recupère les valeurs associées à la caméra
#mtx = matrice de la caméra avec focale + centre optique
#dist = coefficient de distortion
#rvecs = vecteur de rotation
#tvecs = vecteur de translation
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#np.savetxt("data/camDist2.txt", dist)
#np.savetxt("data/camMatrix2.txt", mtx)

"""
img = cv2.imread(images[2])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# undistort
dst = cv2.undistort(img, mtx, dist, None)

# crop the image
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

#Compute error
mean_error = 0
tot_error=0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
    mean_error= tot_error/len(objpoints)

print ("total error: ", mean_error)
"""

