#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:23:40 2019

@author: thomaspaita, quentinpenon
"""

from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2

def main(sweet):
	"""
	sweet: 	0: redbear
		1: greenbear
		2: redcroco
		3: greencroco
		4: carambar
	"""
	####	Prise des photos   ####
	[left, right] = takepicture()	

	####	Detection des objets   ####
	detection(sweet, left, right)
		
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
	print("[INFO] YOLO took {:.6f} seconds for left image".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs_left:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
	 
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
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
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	    

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# thickness of text / boxes
	thickness = 1 + (H+W)//2000

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
	 
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				thickness/4, color, thickness)
	 
	# show the output image
	cv2.imwrite(args["image"][:-4]+"_out.jpg", image)
	cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

		


