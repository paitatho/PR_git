#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:23:40 2019

@author: thomaspaita, quentinpenon
"""

from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
from scipy.optimize import root
import datetime
import imutils
import time
import cv2

def main(sweet):
	"""
	sweet (int) =
        0: redbear 
		1: greenbear 
		2: redcroco
		3: greencroco
		4: carambar
	"""
    
    global arm_len    
    arm_len = [14.5, 18.5, 11]
    global pos
    pos = np.array([0, arm_len[2]])
    
	####	Prise des photos   ####
	[left, right] = takepicture()	

	####	Detection des objets   ####
	detection(sweet, left, right)


    triangulation(left, right, )
    
    # Object position : depth = distance from robot (x-axis) and height (y-axis)
    # We state y = 11 as we want to reach the object from a vertical position
    # in order to remove Theta3 calculus from the reverse cinematic equation
    pos[0] = depth

	####	compute angles from positions   ####
    compute_angles()

	return [theta0, theta1, theta2, theta3]

def takepicture():
	print("[INFO] starting cameras...")
	webcam0 = VideoStream(src=0).start()
	webcam1 = VideoStream(src=1).start()
 	time.sleep(0.5)

	#### 	Capture des images   ####
	frames = []
	for stream in (webcam0, webcam1):
		# read the next frame from the video stream and resize
		frame = stream.read()
		frame = cv2.resize(frame,(640,480))
		frame
		frames.append(frame)
	
	####	correction des distortions    ####
	cameraMatrix = np.loadtxt('data/camMatrix.txt')
	distMatrix = np.loadtxt('data/camDist.txt')
	
	left  = cv2.undistort(frames[0], cameraMatrix, distMatrix, None)
	right = cv2.undistort(frames[1], cameraMatrix, distMatrix, None)
	
	####	libérer stream	  ####
	webcam0.stop()
	webcam1.stop()
	print("[INFO] ending cameras...")
	
	return [left, right]

def detection(sweet):
	print("[INFO] starting detection...")
	folderpath = "Quentin/Application/yolo-sweets"	
	labelsPath = os.path.sep.join(folderpath, "sweets.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	
    confidence_treshold = 0.5
    nms_threshold = 0.3
    
	# derive the paths to the YOLO weights and model configuration
	configPath = os.path.sep.join(folderpath, "sweets-tiny_v4.cfg"])
	weightsPath = os.path.sep.join(folderpath, "sweets-tiny_v4_9400.weights"])
	 
	# load our YOLO object detector trained on sweets dataset (5 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# grab spatial dimensions of our input images
	# image = cv2.imread("imagepath")
	(Hl, Wl) = left.shape[:2]
	(Hr, Wr) = right.shape[:2]
	#print('[INFO] Image Shape : ({},{})'.format(H,W)) 

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
	print("[INFO] YOLO took {:.6f} seconds for left image".format(end - start))

	# right image
	net.setInput(blob_right)
	start = time.time()
	layerOutputs_right = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds for right image".format(end - start))

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
			if classID == sweet and confidence > confidence_treshold:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
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

    # apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 
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
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
	 
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
	 
				# update our list of bounding box coordinates, confidences,
				# and class IDs
                right_centers.append([centerX, centerY])
                right_boxes.append([x, y, int(width), int(height)])
				right_confidences.append(float(confidence))
				right_classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 
                         nms_threshold)


	# thickness of text / boxes
	thickness = 1 + (H+W)//2000

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
	 
			cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				thickness/4, color, thickness)
	 
# =============================================================================
# 	# show the output image
#  	cv2.imwrite(args["image"][:-4]+"_out.jpg", image)
#  	cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#   cv2.imshow("Image", image)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()	
# =============================================================================

# Todo : 
    # try to select the same objects from both detection


		

def func(theta):
    f = [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)) - pos[0], 
         7 + arm_len[0]*np.sin(theta[0]) + arm_len[1]*np.sin(sum(theta)) - pos[1]] 
    
    df = np.array([
        [-arm_len[0]*np.sin(theta[0]) - arm_len[1]*np.sin(sum(theta)), -arm_len[1]*np.sin(sum(theta))], 
        [arm_len[0]*np.cos(theta[0]) + arm_len[1]*np.cos(sum(theta)), arm_len[1]*np.cos(sum(theta))]
        ])

    return f,df


def compute_angles():
    if pos[0] != -1:
        # object exists and its distance from robot is pos[0]
        sol = root(func, [0.5, 0.5], jac=True, method='hybr')
        theta = np.round(sol.x, 3)
        theta3 = - theta[0] + theta[1] + np.pi/2
        # à vérifier ce premeir theta3
        if theta[0] < 0:
            r = np.linalg.norm(pos - np.array([0, 7]))
            alpha = np.arcsin((pos[1]-7)/r)
            theta[0] = - theta[0] + 2*alpha
            theta[1] = - theta[1]
            theta3 = - theta[0] - theta[1] - np.pi/2 
            # <=> theta3 = theta[0] + theta[1] - 2*alpha - np.pi/2  
        theta.append(theta3) # Add calculus of theta3
        return theta 
    else:
        print("[INFO] back to initial position")
        return [0,0] # à définir
    
