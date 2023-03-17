import cv2
import os
import numpy as py
# selecting which camera to access
cap = cv2.VideoCapture(0)
# paths for face & eye cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# setting up video
while True:
    ret, frame = cap.read()
    # making image gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # looping through faces from gray scale frame & making face rectangle and color
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + h, y + w), (255, 0, 0), 5)
        # making eye rectangle and color
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        # looping eye rectangle and color
        for(ex, eh, ew, ey) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    # cap.release()
    # cv2.destroyAllWindows()





