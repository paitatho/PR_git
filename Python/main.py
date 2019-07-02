#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:23:40 2019

@author: thomaspaita, quentinpenon
"""

from __future__ import print_function
from imutils.video import VideoStream
import os
import cv2
import time
import numpy as np
from scipy.optimize import root

def main(sweet, depth):
    """
    sweet (int) =
        0: redbear 
		1: greenbear 
		2: redcroco
		3: greencroco
		4: carambar
    """
    print("   ... python exec ...\n")
    global path
    path = "/home/pi/Desktop/rasp/PR_git/Python/"
    theta = [0]*5
    # theta[0] : indicates if an object has been detected
    # theta[1] : base rotation
    # theta[2] : first arm rotation
    # theta[3] : second arm rotation
    # theta[4] : third arm rotation
    
    global arm_len    
    arm_len = [14.5, 18.5, 11]
    global pos
    pos = np.array([0, arm_len[2]])
    
	####	Caption   ####
    [left, right] = takepicture()
# =============================================================================
#     left = cv2.imread(os.path.sep.join([os.getcwd(),"Quentin/Application/data/leftDisto.png"]))
#     right = cv2.imread(os.path.sep.join([os.getcwd(),"Quentin/Application/data/rightDisto.png"]))
# =============================================================================

	####	Detection and selection of the object   ####
    (x,y), do_shift  = detection(left, right, sweet)
    center = (x,y)
    
    theta[0] = 0 if center == (-1,-1) else 1        
    
    # Object position : depth = distance from robot (x-axis) and height (y-axis)
    # We state y = 11 as we want to reach the object from a vertical position
    # in order to remove Theta3 calculus from the reverse cinematic equation
    
    triangulation(left, right, center)
    pos[0] = depth
    
    #### Left / right shift to center the robot in front of the objet ####
    if do_shift:
        theta[1] = compute_shift(center)
    else:
        theta[1] = 0

	####	compute angles from positions   ####
    theta[2:] = compute_angles()

    # Motor control only understand positive angles
    # besides, they  
    theta[3] = np.pi - abs(theta[3])
    theta[4] = np.pi - abs(theta[4])

    theta[2:] = np.degrees(theta[2:])

    print("[PYTHON] Theta final values : ", list(map(int, theta)))

    return list(map(int, theta))

def takepicture():
    print("[PYTHON] starting cameras...")
    # left_cam
    webcam0 = VideoStream(src=0).start()
    # right_cam
    webcam1 = VideoStream(src=1).start()

    print("[PYTHON] cameras started...")
    time.sleep(0.5)

	#### 	Capture des images   ####
    frames = []
    for stream in (webcam0, webcam1):
        # read the next frame from the video stream and resize
        frame = stream.read()
        frame = cv2.resize(frame,(640,480))
        frames.append(frame)
	
	####	correction des distortions    ####
# =============================================================================
#     cameraMatrix = np.loadtxt('Thomas/data/camMatrix.txt')
#     distMatrix = np.loadtxt('Thomas/data/camDist.txt')
#     left  = cv2.undistort(frames[0], cameraMatrix, distMatrix, None)
#     right = cv2.undistort(frames[1], cameraMatrix, distMatrix, None)
# =============================================================================

    left_cameraMatrix = np.loadtxt(path+'Quentin/Triangulation/camMatrixL.txt')
    left_distMatrix = np.loadtxt(path+'Quentin/Triangulation/camDistL.txt')

    right_cameraMatrix = np.loadtxt(path+'Quentin/Triangulation/camMatrixR.txt')
    right_distMatrix = np.loadtxt(path+'Quentin/Triangulation/camDistR.txt')

    left  = cv2.undistort(frames[0], left_cameraMatrix, left_distMatrix, None)
    right = cv2.undistort(frames[1], right_cameraMatrix, right_distMatrix, None)

    cv2.imwrite(path+'Quentin/Triangulation/data/left_new_undisto.png', left)
    cv2.imwrite(path+'Quentin/Triangulation/data/right_new_undisto.png', right)
    
    ####	libérer stream	  ####
    webcam0.stop()
    webcam1.stop()
    print("[PYTHON] ending cameras...")

    return [left, right]


def triangulation(left, right, center):
    # return depth from the point
    pos[0] = 22

    return 
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
     
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Convert from color(BGR) to gray
    grayL= cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
    grayR= cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(grayR,grayL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    disp = ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

    left_cameraMatrix = np.loadtxt(path+'Quentin/Triangulation/camMatrixL.txt')
    right_cameraMatrix = np.loadtxt(path+'Quentin/Triangulation/camMatrixR.txt')

    # Camera focal
    f = (left_cameraMatrix[0,0]+left_cameraMatrix[1,1])/2.0 + (right_cameraMatrix[0,0]+right_cameraMatrix[1,1])/2.0
    # Distance between the 2 cameras (in millimeters)
    t = 34 

    depthMap = np.zeros(disp.shape)
    mask = disp[:,:] != 0
    depthMap[mask] = f*t / disp[mask]

    cv2.imwrite(path+'Quentin/Triangulation/data/disp_test.png', filteredImg)
    cv2.imwrite(path+'Quentin/Triangulation/data/depthMap_test.png', depthMap)

    return depthMap[center]    


def compute_shift(center):
    # trouver une mesure de la distance de l'objet au centre en mm
    # à partir de la profondeur

    shift = float(abs(320-center[0]))
    sign = 1 if center[0] > 320 else -1
    
    # heuristique forte : dans notre "gamme de distance"
    # on suppose qu'un décalage de 8 pixel équivaut à un décale de 1%

    return sign*shift/8.
    
    
    # sinus(a) = opposé / hypotenus
    #return sign*(np.arcsin(shift/pos[0]))
    
def detection(left, right, sweet):
    print("[PYTHON] starting detection...")
    folderpath = path + "Quentin/Application/yolo-sweets"	
    labelsPath = os.path.sep.join([folderpath, "sweets.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    confidence_threshold = 0.5
    nms_threshold = 0.3

	# derive the paths to the YOLO weights and model configuration
    configPath = os.path.sep.join([folderpath, "sweets-tiny_v4.cfg"])
    weightsPath = os.path.sep.join([folderpath, "sweets-tiny_v4_9400.weights"])
	 
	# load our YOLO object detector trained on sweets dataset (5 classes)
    print("[PYTHON] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# grab spatial dimensions of our input images
	# image = cv2.imread("imagepath")
    (Hl, Wl) = left.shape[:2]
    (Hr, Wr) = right.shape[:2]
	#print('[PYTHON] Image Shape : ({},{})'.format(H,W)) 

	# determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	 
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities

	#left = cv2.resize(left, (640, 480))
    blob_left = cv2.dnn.blobFromImage(left, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	#right = cv2.resize(right, (640, 480))
    blob_right = cv2.dnn.blobFromImage(right, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

	#blob = cv2.dnn.blobFromImage(cv2.resize(image, (416, 416)), 
	#                             1 / 255.0, (416, 416),
	#                             swapRB=True, crop=False)


	# left image
    net.setInput(blob_left)
    start = time.time()
    layerOutputs_left = net.forward(ln)
    end = time.time()
	 
	# show timing information on YOLO
    print("[PYTHON] YOLO took {:.6f} seconds for left image".format(end - start))

	# right image
    net.setInput(blob_right)
    start = time.time()
    layerOutputs_right = net.forward(ln)
    end = time.time()

	# show timing information on YOLO
    print("[PYTHON] YOLO took {:.6f} seconds for right image".format(end - start))

	# initialize our lists of detected bounding boxes, confidences,
	# class IDs, and centers respectively
    left_boxes = []
    left_confidences = []
    left_classIDs = []
    left_centers = []

	# loop over each of the layer outputs
    for output in layerOutputs_left:
		# loop over each of the detections
        for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
	 
            # filter out other objects detections 
			# and weak predictions by ensuring the detected
			# probability is greater than the minimum probability
            if classID == sweet and confidence > confidence_threshold:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
                box = detection[0:4] * np.array([Wl, Hl, Wl, Hl])
                (centerX, centerY, width, height) = box.astype("int")
	 
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
	 
				# update our list of bounding box coordinates, confidences,
				# and class IDs
                left_boxes.append([x, y, int(width), int(height)])
                left_confidences.append(float(confidence))
                left_classIDs.append(classID)
                left_centers.append([centerX, centerY])

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    left_idxs = cv2.dnn.NMSBoxes(left_boxes, left_confidences, confidence_threshold, 
                         nms_threshold)


    # initialize our lists of detected bounding boxes, confidences,
	# class IDs, and centers respectively
    right_boxes = []
    right_confidences = []
    right_classIDs = []
    right_centers = []

    # loop over each of the layer outputs
    for output in layerOutputs_right:
    	# loop over each of the detections
    	for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    	 
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
            if classID == sweet and confidence > confidence_threshold:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
                box = detection[0:4] * np.array([Wr, Hr, Wr, Hr])
                (centerX, centerY, width, height) = box.astype("int")
    	 
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
    
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                right_boxes.append([x, y, int(width), int(height)])
                right_confidences.append(float(confidence))
                right_classIDs.append(classID)
                right_centers.append([centerX, centerY])

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    right_idxs = cv2.dnn.NMSBoxes(right_boxes, right_confidences, confidence_threshold, 
                         nms_threshold)

    if (len(right_idxs) == 0) or (len(left_idxs) == 0):
        return (-1,-1), False

    # If we consider they have found the same objects
    if len(left_idxs.flatten()) == len(right_idxs.flatten()):
        # tri des boxes selon leur valeur de x
        right_x_index = sorted(right_idxs.flatten(), key=lambda k: right_boxes[k][0])
        left_x_index = sorted(left_idxs.flatten(), key=lambda k: left_boxes[k][0])
        # On tri leurs index de manière à travailler sur la même box présumée
        
        # tri des confidences selon leur valeur
        right_conf_index = sorted(right_idxs.flatten(), key=lambda k: right_confidences[k])
        left_conf_index = sorted(left_idxs.flatten(), key=lambda k: left_confidences[k])

        # On veut en prioriter travailler sur l'index dont la confidence est la plus haute
        for k in range(len(right_conf_index)): 
            if right_confidences[right_conf_index[k]] > left_confidences[left_conf_index[k]]:
                max_ind = right_conf_index[k]                    
                xil = left_boxes[max_ind][0]
                xir = left_boxes[max_ind][0] + right_boxes[max_ind][2]
                xjl = right_boxes[max_ind][0]
                xjr = right_boxes[max_ind][0] + right_boxes[max_ind][2]
                # Si même position dans la liste triée selon la position des boites
                # On suppose que ce sont les mêmes objets
                if right_x_index.index(max_ind) == left_x_index.index(max_ind):                    
                    if ((xil+xjl)//2 < 320) and ((xir+xjr)//2 > 320):
                        do_shift = False
                    else:
                        do_shift = True 
                        
                    return ((left_centers[max_ind][0]+right_centers[max_ind][0])//2, 
                            (left_centers[max_ind][1]+right_centers[max_ind][1])//2),do_shift
            else:
                max_ind = left_conf_index[k]
                xil = left_boxes[max_ind][0]
                xir = left_boxes[max_ind][0] + right_boxes[max_ind][2]
                xjl = right_boxes[max_ind][0]
                xjr = right_boxes[max_ind][0] + right_boxes[max_ind][2]

                if right_x_index.index(max_ind) == left_x_index.index(max_ind):
                    if ((xil+xjl)//2 < 320) and ((xir+xjr)//2 > 320):
                        do_shift = False
                    else:
                        do_shift = True 

                    return ((left_centers[max_ind][0]+right_centers[max_ind][0])//2, 
                            (left_centers[max_ind][1]+right_centers[max_ind][1])//2),do_shift
            
# =============================================================================
#     # We consider the object with the higher detection confidence
#     # is the same on the both pictures
#     couple_conf = np.zeros((len(left_idxs.flatten()), len(right_idxs.flatten())))
#     for (n,i) in enumerate(left_idxs.flatten()):
#         for (m,j) in enumerate(right_idxs.flatten()):
#             couple_conf[n][m] = left_confidences[n] + right_confidences[m]
# 
#     # retrieving indices of the minimum distance value
#     max_conf = np.where(couple_conf == np.amin(couple_conf))
#     max_conf = list(zip(max_conf[0], max_conf[1]))
# 
#     # we just take the first one    
#     for (n,m) in max_conf:
#         # Middle from ( [xA, yA] , [xB,yB] )
#         return ((left_centers[left_idxs.flatten()[n]][0]+right_centers[right_idxs.flatten()[m]][0])//2, 
#                 (left_centers[left_idxs.flatten()[n]][1]+right_centers[right_idxs.flatten()[m]][1])//2)        
# 
# =============================================================================

    # backup technique ? take the closest boxes ?
    dist = np.zeros((len(left_idxs.flatten()), len(right_idxs.flatten())))
    for (n,i) in enumerate(left_idxs.flatten()):
        xi, yi = left_boxes[i][0], left_boxes[i][1]
        wi, hi = left_boxes[i][2], left_boxes[i][3]
        for (m,j) in enumerate(right_idxs.flatten()):
            xj, yj = right_boxes[j][0], right_boxes[j][1]
            wj, hj = right_boxes[j][2], right_boxes[j][3]
            
            dist[n][m] = np.sqrt(np.power(xj - xi, 2) + np.power(yj - yi, 2))

    # retrieving indices of the minimum distance value
    min_dist = np.where(dist == np.amin(dist))
    min_dist = list(zip(min_dist[0], min_dist[1]))

    # we just take the first one    
    for (n,m) in min_dist:
        # Middle from ( [xA, yA] , [xB,yB] )
        return ((left_centers[left_idxs.flatten()[n]][0]+right_centers[right_idxs.flatten()[m]][0])//2, 
                (left_centers[left_idxs.flatten()[n]][1]+right_centers[right_idxs.flatten()[m]][1])//2)


    # we consider they have found the same objects
    if len(left_idxs.flatten()) == len(right_idxs.flatten()):
        # vertical distance from two boxes
        dist = np.zeros((len(left_idxs.flatten()), len(right_idxs.flatten())))
        for (n,i) in enumerate(left_idxs.flatten()):
            xi, yi = left_boxes[i][0], left_boxes[i][1]
            wi, hi = left_boxes[i][2], left_boxes[i][3]
            for (m,j) in enumerate(right_idxs.flatten()):
                xj, yj = right_boxes[j][0], right_boxes[j][1]
                wj, hj = right_boxes[j][2], right_boxes[j][3]
                
                dist[n][m] = np.sqrt(np.power(xj - xi, 2) + np.power(yj - yi, 2))

        # retrieving indices of the minimum distance value
        min_dist = np.where(dist == np.amin(dist))
        min_dist = list(zip(min_dist[0], min_dist[1]))

        # we just take the first one    
        for (n,m) in min_dist:
            # Middle from ( [xA, yA] , [xB,yB] )
            return ((left_centers[left_idxs.flatten()[n]][0]+right_centers[right_idxs.flatten()[m]][0])//2, 
                    (left_centers[left_idxs.flatten()[n]][1]+right_centers[right_idxs.flatten()[m]][1])//2)
    else:
        # compute Intersection over Union between the predicted boxes
        # to try to match a left and right object
        iou = np.zeros((len(left_idxs.flatten()), len(right_idxs.flatten())))
        for (n,i) in enumerate(left_idxs.flatten()):
            xi, yi = left_boxes[i][0], left_boxes[i][1]
            wi, hi = left_boxes[i][2], left_boxes[i][3]
            for (m,j) in enumerate(right_idxs.flatten()):
                xj, yj = right_boxes[j][0], right_boxes[j][1]
                wj, hj = right_boxes[j][2], right_boxes[j][3]
    
                # define intersection coordonates of the two boxes             
                xA = max(xi,xj) 
                yA = max(yi,yj)
                xB = min(xi+wi, xj+wj)
                yB = min(yi+hi, yj+hj)
                
            	# compute t1he area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
             
            	# compute the area of both boxes left and right
                boxAArea = (wi + 1) * (hi + 1)
                boxBArea = (wj + 1) * (hj + 1)
             
            	# compute the intersection over union by taking the intersection
            	# area and dividing it by the sum of prediction + ground-truth
            	# areas - the interesection area
                iou[n][m] = interArea / float(boxAArea + boxBArea - interArea)
             
            	
        # retrieving indices of the maximum IoU value
        res = np.where(iou == np.amax(iou))
        res = list(zip(res[0], res[1]))
    
        # we just take the first one    
        for (n,m) in res:
            print(n,m)
            # Middle from ( [xA, yA] , [xB,yB] )
            return ((left_centers[left_idxs[n]][0]+right_centers[right_idxs[m]][0])//2, 
                    (left_centers[left_idxs[n]][1]+right_centers[right_idxs[m]][1])//2)
        
		

def func(theta):
    # The base of the robot arm is about 7 cm raised 
    f = [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)) - pos[0], 
         7 + arm_len[0]*np.sin(theta[0]) + arm_len[1]*np.sin(sum(theta)) - pos[1]] 
    
    df = np.array([
        [-arm_len[0]*np.sin(theta[0]) - arm_len[1]*np.sin(sum(theta)), -arm_len[1]*np.sin(sum(theta))], 
        [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)), arm_len[1]*np.cos(sum(theta))]
        ])

    return f,df


def compute_angles():
    if pos[0] != -1:
        print("[PYTHON] Reverse Cinematic ... ")
        # object exists and its distance from robot is pos[0]
        sol = root(func, [0.5, 0.5], jac=True, method='hybr')        
        theta = np.around(sol.x, 3).tolist()
        theta3 = - theta[0] + theta[1] + np.pi/2
        # à vérifier ce premeir theta3
        if theta[0] < 0:
            r = np.linalg.norm(pos - np.array([0, 7]))
            alpha = np.arcsin((pos[1]-7)/r)
            theta[0] = - theta[0] + 2*alpha
            theta[1] = - theta[1]
            theta3 = - theta[0] - theta[1] - np.pi/2 
            # <=> theta3 = theta[0] + theta[1] - 2*alpha - np.pi/2  
        # Add calculus of theta3
        theta.append(theta3)
        
        return theta
    else:
        print("[PYTHON] back to initial position")
        return [0,0,0] # à définir

