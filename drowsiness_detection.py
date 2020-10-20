# to run:- python drowsiness_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat

# Import the necessary packages
import config
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from twilio.rest import Client
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from scipy.spatial import distance as dist
from playsound import playsound
from matplotlib import style 
from datetime import datetime
import os 
import csv
import numpy as np
import pandas as pd
import imutils 
import dlib
import time 
import argparse 
import cv2 


#client to send SMS to mobile device
client = Client(config.SSID, config.AUTH_TOKEN)

style.use('fivethirtyeight')
# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


#all EAR and MAR with time
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]
# Parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type = int, default = -1, help = "whether raspberry pi camera shall be used or not")
args = vars(ap.parse_args())

# EAR threshold value, below which a blink is considered
EAR_THRESHOLD = 0.25
# Number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 15 
# MAR threshold value
MAR_THRESHOLD = 21

# Counters inititalization 
BLINK_COUNT = 0 
FRAME_COUNT = 0 

# Intializing the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO]Loading the predictor.....")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])

# Extract indexes of left eye, right eye and mouth respectively
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Start the video stream from the camera
print("[INFO]Loading Camera.....")
vs = VideoStream(usePiCamera = args["picamera"] > 0).start()
time.sleep(2) 

assure_path_exists("dataset/")

# Counters to count sleep and yawn in the video stream
count_sleep = 0
count_yawn = 0 

 
# Loop over the video stream frame by frame
while True: 
	# Frame extraction
	frame = vs.read()
	cv2.putText(frame, "PRESS 'e' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 
	# Resizing of the frame 
	frame = imutils.resize(frame, width = 500)
	# Grayscale conversion of the frame 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Face detection by the dlib's detector
	rects = detector(frame, 1)

	# Predict facial landmarks on the face detected by the detector 
	for (i, rect) in enumerate(rects): 
		shape = predictor(gray, rect)
		# Convert it to a (68, 2) size numpy array 
		shape = face_utils.shape_to_np(shape)

		# Bounding box for the detected face 
		(x, y, w, h) = face_utils.rect_to_bb(rect) 
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
		cv2.putText(frame, "DRIVER", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		leftEye = shape[lstart:lend]
		rightEye = shape[rstart:rend] 
		mouth = shape[mstart:mend]
		# Calculation of the EAR for both the eyes 
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Taking average of the EAR's of both eyes
		EAR = (leftEAR + rightEAR) / 2.0
		# Updation of value in a .csv file
		ear_list.append(EAR)
		#print(ear_list)
		

		ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
		# Compute and visulaize the convex hull for both the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# Draw the contours on both the eyes................... 
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

		MAR = mouth_aspect_ratio(mouth)
		mar_list.append(MAR/10)
		# Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
		# Thus, count the number of frames for which the eye remains closed 
		if EAR < EAR_THRESHOLD: 
			FRAME_COUNT += 1

			cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

			if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
				count_sleep += 1
				# Add the frame to the dataset ar a proof of drowsy driving
				cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
				playsound('sound files/alarm.mp3')
				cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				if count_sleep%3 ==0:
                                    message = client.messages.create(to=config.TO_NUMBER , from_=config.FROM_NUMBER , body="Warning!! Driver of vehicle number:PB XX 01 XXX, has been sensed as drowsy "+str(count_sleep)+" times. You are advised to take appropriate actions to avoid accident")
                                    print(message.sid)
                                    
		else: 
			if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
				playsound('sound files/warning.mp3')
			FRAME_COUNT = 0
		#cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Check if the person is yawning
		if MAR > MAR_THRESHOLD:
			count_yawn += 1
			cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
			cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# Add the frame to the dataset ar a proof of drowsy driving
			cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
			playsound('sound files/alarm.mp3')
			playsound('sound files/warning_yawn.mp3')
			if count_yawn%3 ==0:
                            message = client.messages.create(to=config.TO_NUMBER , from_=config.FROM_NUMBER , body="Warning!! Driver of vehicle number:PB XX 01 XXX, has been continuously yawning. You are advised to take appropriate actions to avoid accident")
                            print(message.sid)
	#total data collection for plotting
	for i in ear_list:
		total_ear.append(i)
	for i in mar_list:
		total_mar.append(i)			
	for i in ts:
		total_ts.append(i)
	#display the frame 
	cv2.imshow("Output", frame)
	key = cv2.waitKey(1) & 0xFF 
	
	

	if key == ord('e'):
		break

a = total_ear
b=total_mar
c = total_ts

df = pd.DataFrame({"EAR" : a, "MAR":b,"TIME" : c})
df.to_csv("op_webcam.csv", index=False)

cv2.destroyAllWindows()
vs.stop()
