#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#      ▄▀▄     ▄▀▄
#     ▄█░░▀▀▀▀▀░░█▄
# ▄▄  █░░░░░░░░░░░█  ▄▄
#█▄▄█ █░░▀░░┬░░▀░░█ █▄▄█

###################################
##### Authors:                #####
##### Stephane Vujasinovic    #####
##### Frederic Uhrweiller     ##### 
#####                         #####
##### Creation: 2017          #####
###################################


#***********************
#**** Main Programm ****
#***********************


# Package importation
import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize

# Filtering
kernel= np.ones((3,3),np.uint8)

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        
        print('Distance: '+ str(Distance)+' m')
        
# This section has to be uncommented if you want to take mesurements and store them in the excel
##        ws.append([counterdist, average])
##        print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
##        if (counterdist <= 85):
##            counterdist += 3
##        elif(counterdist <= 120):
##            counterdist += 5
##        else:
##            counterdist += 10
##        print('Next distance to measure: '+str(counterdist)+'cm')

# Mouseclick callback
wb=Workbook()
ws=wb.active  

#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************

Left_Stereo_Map_0 = np.loadtxt('Left_Stereo_Map_0.txt')
Left_Stereo_Map_0 = Left_Stereo_Map_0.reshape((480, 640, 2))
Left_Stereo_Map_1 = np.loadtxt('Left_Stereo_Map_1.txt')

Right_Stereo_Map_0 = np.loadtxt('Right_Stereo_Map_0.txt')
Right_Stereo_Map_0 = Right_Stereo_Map_0.reshape((480, 640, 2))
Right_Stereo_Map_1 = np.loadtxt('Right_Stereo_Map_1.txt')


frameL= cv2.imread("data/left0.png")
frameR= cv2.imread("data/right0.png")

# Rectify the images on rotation and alignement
Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

##    # Draw Red lines
##    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
##        Left_nice[line*20,:]= (0,0,255)
##        Right_nice[line*20,:]= (0,0,255)
##
##    for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
##        frameL[line*20,:]= (0,255,0)
##        frameR[line*20,:]= (0,255,0)    
    
# Show the Undistorted images
cv2.imwrite('data/Right_nice.png', Right_nice)
cv2.imwrite('data/Left_nice.png', Left_nice)

# Convert from color(BGR) to gray
grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)

# Compute the 2 images for the Depth_image
disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
dispL= disp
dispR= stereoR.compute(grayR,grayL)
dispL= np.int16(dispL)
dispR= np.int16(dispR)

# Using the WLS filter
filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
#cv2.imshow('Disparity Map', filteredImg)
disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

# =============================================================================
#     f = (cameraMatrix[0,0]+cameraMatrix[1,1]) / 2 # focale de la caméra
#     t = 34 
# 
#     depthMap = np.zeros(disparity.shape)
#     mask = disparity[:,:] != 0
#     depthMap[mask] = f*t /disparity[mask] 
# =============================================================================

##    # Resize the image for faster executions
##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

# Filtering the Results with a closing filter
closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

# Colors map
dispc= (closing-closing.min())*255
dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

# Show the result for the Depth_image
cv2.imwrite('data/Disparity.png', disp)
cv2.imwrite('data/Closing.png',closing)
cv2.imwrite('data/Color_Depth.png',disp_Color)
cv2.imwrite('data/Filtered_Color_Depth.png',filt_Color)

# Mouse click
#cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)

# End the Programme
#if cv2.waitKey(1) & 0xFF == ord(' '):
#    cv2.destroyAllWindows()


# =============================================================================
# # Call the two cameras
# CamR= cv2.VideoCapture(1)   # When 1 then Right Cam and when 2 Left Cam
# CamL= cv2.VideoCapture(2)
# 
# 
# 
# while True:
#     # Start Reading Camera images
#     retR, frameR= CamR.read()
#     retL, frameL= CamL.read()
# 
# 
#     #frameL= cv2.imread("/home/quentin/Documents/UTC/P2019/PR_Robot/PR_git/Python/Thomas/data/left0.png")
#     #frameR= cv2.imread("/home/quentin/Documents/UTC/P2019/PR_Robot/PR_git/Python/Thomas/data/right0.png")
# 
# 
#     # Rectify the images on rotation and alignement
#     Left_nice= cv2.remap(frameL,Left_Stereo_Map_0,Left_Stereo_Map_1, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
#     Right_nice= cv2.remap(frameR,Right_Stereo_Map_0,Right_Stereo_Map_1, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
# 
# ##    # Draw Red lines
# ##    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
# ##        Left_nice[line*20,:]= (0,0,255)
# ##        Right_nice[line*20,:]= (0,0,255)
# ##
# ##    for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
# ##        frameL[line*20,:]= (0,255,0)
# ##        frameR[line*20,:]= (0,255,0)    
#         
#     # Show the Undistorted images
#     #cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
#     #cv2.imshow('Normal', np.hstack([frameL, frameR]))
# 
#     # Convert from color(BGR) to gray
#     grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
#     grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
# 
#     # Compute the 2 images for the Depth_image
#     disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
#     dispL= disp
#     dispR= stereoR.compute(grayR,grayL)
#     dispL= np.int16(dispL)
#     dispR= np.int16(dispR)
# 
#     # Using the WLS filter
#     filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
#     filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#     filteredImg = np.uint8(filteredImg)
#     #cv2.imshow('Disparity Map', filteredImg)
#     disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
# 
# ##    # Resize the image for faster executions
# ##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)
# 
#     # Filtering the Results with a closing filter
#     closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 
# 
#     # Colors map
#     dispc= (closing-closing.min())*255
#     dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
#     disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
#     filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 
# 
#     # Show the result for the Depth_image
#     #cv2.imshow('Disparity', disp)
#     #cv2.imshow('Closing',closing)
#     #cv2.imshow('Color Depth',disp_Color)
#     cv2.imshow('Filtered Color Depth',filt_Color)
# 
#     # Mouse click
#     cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
#     
#     # End the Programme
#     if cv2.waitKey(1) & 0xFF == ord(' '):
#         break
#     
# # Save excel
# ##wb.save("data4.xlsx")
# 
# # Release the Cameras
# CamR.release()
# CamL.release()
# cv2.destroyAllWindows()
# 
# =============================================================================
