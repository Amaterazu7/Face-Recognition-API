import numpy as np
import cv2
import pickle
import string
import random

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
project_path = "F:/FUNDAMENTOS/images/"
folder_data_traning = "alan-wieilly/"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	font = cv2.FONT_HERSHEY_SIMPLEX
    	color = (255, 255, 255)
    	stroke = 2
    	if conf >= 40 and conf <= 85:
    		name = labels[id_]
    		printString = '%s %.2f %%' % (name, conf)
    		#cv2.putText(frame, printString.replace("-"," "), (0,30), font, 1, color, stroke, cv2.LINE_AA)
    		cv2.putText(frame, printString.replace("-"," "), (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		#cv2.putText(frame, "Pertenece al curso", (0,470), font, 1, color, stroke, cv2.LINE_AA)
    	else:
    		cv2.putText(frame, "No pertenece", (x,y), font, 1, color, stroke, cv2.LINE_AA)
    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    if cv2.waitKey(20) & 0xFF == ord('w'):
    	img_item = project_path + folder_data + random.choice(string.letters) + ".png"
    	#img_item = "a.png"
    	cv2.imwrite(img_item,roi_color)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
