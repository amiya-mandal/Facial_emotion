import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('harr_cass.xml')
cap = cv2.VideoCapture(0)

count = 0 
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            count += 1
            print ('face detected')
            cv2.imshow('frame',frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else: 
            break

print (count)