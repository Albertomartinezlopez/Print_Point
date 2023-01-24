"""
__author__ = "Alberto"
__version__ = "2023.01.24"
"""

# coding: utf-8

# In[1]:

import cv2
import numpy as np

image = cv2.imread("lines.png")

while True:
    ret, orig_frame = image.read()
    if not ret:
        image = cv2.ImageCapture("lines.png")
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BRG2GRAY)
    low_red = np.array([13,100,100])
    up_red = np.array([337,100,100])
    mask = cv2.inRange(hsv, low_red, up_red)
    edges = cv2.Canny(mask, 75,150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line [0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)



    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    key = cv2.WaitKey(25)
    if key == 27:
        break
image.release()
cv2.destroyAllWindows()