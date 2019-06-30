#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if "Tkinter" not in sys.modules:
    from tkinter import *
import cv2 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.preprocessing import normalize
from tkinter import *
from PIL import Image, ImageTk 
import tkinter.messagebox
import tkinter.filedialog

window_size = 3                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,             # max_disp has to be dividable by 16 f. E. HH 192, 256
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



pathL = "data/leftDisto.png"
pathR = "data/rightDisto.png"
pathL = "data/left1.jpg"
pathR = "data/right1.jpg"

def getParam():
    
    window_size = 3                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
     
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities= numDisp.get(),             # max_disp has to be dividable by 16 f. E. HH 192, 256
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

def getDisparity(imgL, imgR,n=1):
    # SGBM Parameters -----------------
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

def Ouvrir():
    Canevas.delete(ALL) # on efface la zone graphique

    filename = tkinter.filedialog.askopenfilename(title="Ouvrir une image",filetypes=[('gif files','.gif'),('all files','.*')])
    print(filename)

    photo = PhotoImage(file=filename)
    gifdict[filename] = photo  # référence
    print(gifdict)

    Canevas.create_image(0,0,anchor=NW,image=photo)
    Canevas.config(height=photo.height(),width=photo.width())

    Mafenetre.title("Image "+str(photo.width())+" x "+str(photo.height()))

def Fermer():
    Canevas.delete(ALL)
    Mafenetre.title("Image")

def Apropos():
    tkinter.messagebox.showinfo("A propos","Tutorial Python Tkinter\n(C) Fabrice Sincère")


cameraMatrix = np.loadtxt('data/camMatrix.txt')
distMatrix = np.loadtxt('data/camDist.txt')

f = (cameraMatrix[0,0]+cameraMatrix[1,1]) / 2 # focale de la caméra
t = 34 

def computeDisparity(c):
    #print("Value: ",numDisp.get())
    imgL = cv2.imread(pathL,0)
    imgR = cv2.imread(pathR,0)

    window_size = blk.get()                   # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=minDisp.get(),
        numDisparities= numDisp.get(),             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=blk.get(),
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=maxDiff.get(),
        speckleWindowSize=speckWin.get(),
        speckleRange=speckRange.get(),
        preFilterCap=63,
        uniquenessRatio=uni.get(),
        #mode=cv2.STEREO_SGBM_MODE_HH 
        #mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        mode=mode.get()
    ) 
    
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
     
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
     
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
     
    print('computing disparity 1 ...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
     
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    
    cv2.imwrite('data/dispTmp.png',filteredImg)
    photoDisp= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/dispTmp.png")
    Canevas.create_image(photoDisp.width()+10,photoDisp.height()+10,anchor=NW,image=photoDisp) 
    
    depthMap = np.zeros(filteredImg.shape)
    mask = filteredImg[:,:] != 0
    depthMap[mask] = f*t /filteredImg[mask] 
    cv2.imwrite('data/depthTmp.png',depthMap)
    
    photoDepth= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/depthTmp.png")
    Canevas.create_image(0,photoDepth.height()+10,anchor=NW,image=photoDepth) 
    
    Tk.update()


def computeDisparity(c):
    print('computing disparity 2 ...')
    imgL = cv2.imread(pathL,0)
    imgR = cv2.imread(pathR,0)
    window_size = blk.get()
    stereo = cv2.StereoSGBM_create(minDisparity=minDisp.get(),
                                   numDisparities=numDisp.get(), 
                                   blockSize=blk.get(),
                                   mode=mode.get(),
                                   speckleWindowSize=speckWin.get(),
                                   speckleRange=speckRange.get(),        
                                   disp12MaxDiff=maxDiff.get(),
                                   uniquenessRatio=uni.get(),
                                   preFilterCap=63,
                                   P1=8 * 2 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                                   P2=32 * 2 * window_size ** 2,
                                   )
    
    disparity = stereo.compute(imgL ,imgR)
    disparity = disparity/16

    cv2.imwrite('data/dispTmp.png',disparity)
    photoDisp= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/dispTmp.png")
    Canevas.create_image(0,photoDisp.height()+10,anchor=NW,image=photoDisp) 
    
    depthMap = np.zeros(disparity.shape)
    mask = disparity[:,:] != 0
    depthMap[mask] = f*t /disparity[mask] 
    cv2.imwrite('data/depthTmp.png',depthMap)
    
    photoDepth= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/depthTmp.png")
    Canevas.create_image(photoDepth.width() +10,photoDepth.height()+10,anchor=NW,image=photoDepth) 
    
    Tk.update()
    
    

# Main window
Mafenetre = Tk()
Mafenetre.title("Image")

# Affichage du menu
Mafenetre.grid_rowconfigure(0, weight=1)
Mafenetre.grid_columnconfigure(0, weight=1)

# Création d'un widget Canvas
photo= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/leftDisto.png")
Canevas = Canvas(Mafenetre)
Canevas.pack(padx=5,pady=5)
Canevas.create_image(0,0,anchor=NW,image=photo)
Canevas.config(height=photo.height(),width=photo.width())

photo2= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/rightDisto.png")
Canevas.create_image(photo2.width()+10,0,anchor=NW,image=photo2)

left = cv2.imread(pathL,0)
right = cv2.imread(pathR,0)
disparity = getDisparity(left,right,1)
cv2.imwrite('data/dispTmp.png',disparity)

depthMap = np.zeros(disparity.shape)
mask = disparity[:,:] != 0
depthMap[mask] = f*t /disparity[mask] 
cv2.imwrite('data/depthTmp.png',depthMap)

photoDisp= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/dispTmp.png")
Canevas.create_image(0,photoDisp.height()+10,anchor=NW,image=photoDisp)   
photoDepth= PhotoImage(file="/home/thomaspaita/Bureau/General/Cours/GI04/PR_Bras_Robot/PR_git/Python/Thomas/data/depthTmp.png")
Canevas.create_image(photoDepth.width() +10,photoDepth.height()+10,anchor=NW,image=photoDepth) 

Canevas.config(height=10+2*photo2.height(),width=photo.width()+photo2.width())

Canevas.grid(row=0, column=0, sticky="nsew")

# minDisp.config( command = computeDisparity  )
numDisp = Scale(Mafenetre, orient='vertical', from_=16, to=16*20,
      resolution=16, tickinterval=2, length=350,
      label='numDisp',command=computeDisparity)
numDisp.grid(row=0, column=1)

minDisp = Scale(Mafenetre, orient='vertical', from_=0, to=16*3,
      resolution=2, tickinterval=2, length=350,
      label='minDisp',command=computeDisparity)
minDisp.grid(row=0, column=2)

blk = Scale(Mafenetre, orient='vertical', from_=1, to=30,
      resolution=1, tickinterval=1, length=350,
      label='blkSize',command=computeDisparity)
blk.grid(row=0, column=3)

# =============================================================================
# win = Scale(Mafenetre, orient='vertical', from_=3, to=7,
#       resolution=2, tickinterval=1, length=350,
#       label='winSize',command=computeDisparity)
# win.grid(row=0, column=4)
# =============================================================================

mode = Scale(Mafenetre, orient='vertical', from_=0, to=3,
      resolution=1, tickinterval=1, length=100,
      label='mode',command=computeDisparity)
mode.grid(row=0, column=5)

speckRange = Scale(Mafenetre, orient='vertical', from_=0, to=30,
      resolution=1, tickinterval=1, length=350,
      label='speckle',command=computeDisparity)
speckRange.grid(row=0, column=6)

speckWin = Scale(Mafenetre, orient='vertical', from_=50, to=200,
      resolution=10, tickinterval=1, length=350,
      label='speckleWin',command=computeDisparity)
speckWin.grid(row=0, column=7)

maxDiff = Scale(Mafenetre, orient='vertical', from_=0, to=50,
      resolution=2, tickinterval=1, length=350,
      label='maxDiff',command=computeDisparity)
maxDiff.grid(row=0, column=8)

uni = Scale(Mafenetre, orient='vertical', from_=5, to=15,
      resolution=2, tickinterval=1, length=350,
      label='uniquess',command=computeDisparity)
uni.grid(row=0, column=9)
# Utilisation d'un dictionnaire pour conserver une référence
gifdict={}

Mafenetre.mainloop()


