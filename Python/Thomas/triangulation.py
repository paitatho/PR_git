#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:23:40 2019

@author: thomaspaita
"""

import cv2 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
from sklearn.preprocessing import normalize

############# fonctions

def ptImage(image):
    plt.figure()
    plt.xticks(np.arange(0, image.shape[1], 50))
    plt.yticks(np.arange(0, image.shape[0], 50))
    plt.imshow(image,'gray')
    #plt.savefig('data/depth.png')

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

#square : ligne, colonne, hauteur, largeur
def drawSquare(im,squares,color='r'):
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    #                  Rectangle((j,i),largeur,hauteur,linewidth=1,edgecolor='r',facecolor='none')
    for i in range(len(squares)):
        rect = patches.Rectangle((squares[i][1],squares[i][0]),squares[i][3],squares[i][2],linewidth=2,edgecolor=color,facecolor='none')
        ax.add_patch(rect)
    plt.show()
 
def getDisparity(imgL, imgR,n=1):
    # SGBM Parameters -----------------
    window_size = 3                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
     
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*n,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=-1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        #mode=cv2.STEREO_SGBM_MODE_HH 
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



# disparity settings
window_size = 5
min_disp = 32
num_disp = 112-min_disp
stereo = cv2.StereoSGBM(
    minDisparity = min_disp,
    numDisparities = num_disp,
    SADWindowSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
)
 
# morphology settings
kernel = np.ones((12,12),np.uint8)
 
disparity = stereo.compute(left,right).astype(np.float32) / 16.0
disparity = (disparity-min_disp)/num_disp

threshold = cv2.threshold(disparity, 0.6, 1.0, cv2.THRESH_BINARY)[1]
############# Fin fonctions




#récupére les caractéristiques de la caméra calculé dans calibration.py

cameraMatrix = np.loadtxt('data/camMatrix.txt')
distMatrix = np.loadtxt('data/camDist.txt')

f = (cameraMatrix[0,0]+cameraMatrix[1,1]) / 2 # focale de la caméra
t = 34 # distance entre les deux caméras en mm

#charge image de gauche et de droite
left = cv2.imread("data/left0.png",0)
right = cv2.imread("data/right0.png",0)
printImages([left,right], 1,2)

left = cv2.undistort(left, cameraMatrix, distMatrix, None)
right = cv2.undistort(right, cameraMatrix, distMatrix, None)

printImages([left,right], 1,2)

# FACON 1 DE CALCULER MAP DES DISPARRITES
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=7*7)
disparity = stereo.compute(left,right)
ptImage(disparity)

# FACON 2 DE CALCULER MAP DES DISPARRITES
disparity = getDisparity(left,right,1)
ptImage(disparity)
#drawSquare(disparity,[[400,300,20,20]])

#CALCUL MATRICE DES PROFONDEURS PAR PIXEL
depthMap = np.zeros(disparity.shape)
#comme il faut diviser par la disparité on mask les cases à 0
mask = disparity[:,:] != 0
depthMap[mask] = f*t /disparity[mask] 
ptImage(depthMap)








