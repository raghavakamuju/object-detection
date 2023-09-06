#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("color")
cv2.createTrackbar("h", "color", 0, 255, nothing)
cv2.createTrackbar("s", "color", 0, 255, nothing) 
cv2.createTrackbar("v", "color", 0, 255, nothing)  
cv2.createTrackbar("h1", "color", 0, 255, nothing)
cv2.createTrackbar("s1", "color", 0, 255, nothing)
cv2.createTrackbar("v1", "color", 0, 255, nothing)

while True:
    ret, frame = cap.read()
    
    h_l = cv2.getTrackbarPos("h", "color")
    s_l = cv2.getTrackbarPos("s", "color")
    v_l = cv2.getTrackbarPos("v", "color")
    h_h = cv2.getTrackbarPos("h1", "color")
    s_h = cv2.getTrackbarPos("s1", "color")
    v_h = cv2.getTrackbarPos("v1", "color")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([h_l, s_l, v_l])
    upper_bound = np.array([h_h, s_h, v_h])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow("result", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# In[ ]:




