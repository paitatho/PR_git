#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:23:40 2019

@author: thomaspaita
"""

import cv2 
from matplotlib import pyplot as plt
import sys
import numpy as np
from sklearn.preprocessing import normalize

############# fonctions

def ptImage(image):
    plt.figure()
    plt.axis("off")
    plt.imshow(image,'gray')

def printImages(images, nbL, nbC):
    plt.figure(1)
   
    for i, image in enumerate(images):
        plt.subplot(nbL,nbC, i+1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def square(image):
    cv2.rectangle(image,(384,0),(510,128),(0,255,0),3)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
def getDisparity(imgL, imgR):
    # SGBM Parameters -----------------
    window_size = 5                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
     
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
     
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
     
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
     
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
     
    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
     
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
# =============================================================================
#     cv2.imshow('Disparity Map', filteredImg)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
# =============================================================================
    return filteredImg

############# Fin fonctions




#récupére les caractéristiques de la caméra calculé dans calibration.py
cameraMatrix = np.loadtxt('data/webcamMatrix.txt')
f = (cameraMatrix[0,0]+cameraMatrix[1,1]) / 2 # focale de la caméra
t = 40 # distance entre les deux caméras 4cm

#charge image de gauche et de droite
left = cv2.imread("data/left.jpg",0)
right = cv2.imread("data/right.jpg",0)
printImages([left,right], 1,2)

# FACON 1 DE CALCULER MAP DES DISPARRITES
stereo = cv2.StereoSGBM_create(numDisparities=16*2, blockSize=15)
disparity = stereo.compute(left,right)

# FACON 2 DE CALCULER MAP DES DISPARRITES
disparity = getDisparity(left,right)

ptImage(disparity)

#CALCUL MATRICE DES PROFONDEURS PAR PIXEL
depthMap = np.zeros(disparity.shape)
#comme il faut diviser par la disparité on mask les cases à 0
mask = disparity[:,:] != 0
depthMap[mask] = f*t /disparity[mask] 
ptImage(depthMap)







