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
webcam0 = VideoStream(src=0).start()
webcam1 = VideoStream(src=1).start()
time.sleep(1.0)
 
# number of frames read
total = 0
nb=0
# loop over frames from the video streams
while True:
	# initialize the list of frames that have been processed
	frames = []
 
	# loop over the frames and their respective motion detectors
	for stream in (webcam0, webcam1):
		# read the next frame from the video stream and resize
		# it to have a maximum width of 400 pixels
		frame = stream.read()
		frame = cv2.resize(frame, (640,480))
		frame = np.flipud(frame)
		frames.append(frame)

	# increment the total number of frames read and grab the 
	# current timestamp
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
 
	# loop over the frames a second time
    for (frame, name) in zip(frames, ("Webcam0", "Webcam1")):
		# draw the timestamp on the frame and display it
		cv2.imshow(name, frame)
	
	# check to see if a key was pressed
	key = cv2.waitKey(1) # & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	elif key == ord("t"):
		string="/home/pi/Bureau/PR_git/Python/Thomas/data/" + "right" + str(nb) + ".png"
		string1="/home/pi/Bureau/PR_git/Python/Thomas/data/" + "left" + str(nb) + ".png"
		print("image taked\n")
		cv2.imwrite(string,frames[0])
		cv2.imwrite(string1,frames[1])
		nb+=1
		

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
webcam0.stop()
webcam1.stop()
