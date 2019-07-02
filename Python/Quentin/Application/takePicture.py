# import the necessary packages
from __future__ import print_function
#from pyimagesearch.basicmotiondetector import BasicMotionDetector
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2
 
# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
webcam0 = VideoStream(src=1).start()
webcam1 = VideoStream(src=2).start()
time.sleep(2.0)
 
# number of frames read
total = 0

# loop over frames from the video streams
while True:
	# initialize the list of frames that have been processed
	frames = []
 
	# loop over the frames and their respective motion detectors
	for stream in (webcam0, webcam1):
		# read the next frame from the video stream and resize
		frame = stream.read()
		frame = imutils.resize(frame, width=480)
 
		# convert the frame to grayscale, blur it slightly, update
		# the motion detector
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
		# we should allow the motion detector to "run" for a bit
		# and accumulate a set of frames to form a nice average
		if total < 32:
			frames.append(frame)
			continue			
			# loop over frames from the video streams

		frames.append(frame)

	# increment the total number of frames read and grab the 
	# current timestamp
	total += 1
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
 
	# loop over the frames a second time
	for (frame, name) in zip(frames, ("Webcam0", "Webcam1")):
        # draw the timestamp on the frame and display it
        cv2.imshow(name, frame)
        cv2.imwrite(name, frame)
        
    # check to see if a key was pressed
	key = cv2.waitKey(1) # & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
    elif key == ord(" "):
        cv2.imwrite(name, frame)
        
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
webcam0.stop()
webcam1.stop()
