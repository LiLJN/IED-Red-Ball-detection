import cv2
import time
import numpy as np

red_Low = np.array([169, 99, 99]) 
red_Up = np.array([178, 255, 255])

capture=cv2.VideoCapture(0)

flag_red_ball = 0
while(True):
    flag_red_ball = 0
    ret,frame=capture.read()
    if not ret:
        print('Capture Failed！')
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, red_Low, red_Up)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centre = None
    if len(cnts) > 0:
        c = max(cnts, key = cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        centre = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        if radius > 10:
            flag_red_ball = 1
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, centre, 3, (0, 0, 255), -1)
    cv2.imshow('Frame', frame)
    k = cv2.waitKey(5)&0xFF
    if k == 27:
        break
capture.release()
cv2.destroyAllWindows()
