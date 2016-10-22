import cv2 as cv
import numpy as np
import matplotlib.pyplot as mpl

cap = cv.VideoCapture(0)

#img = cv.imread('img.jpg', 0)

#cv.imshow('title', img)
#cv.waitKey(0)
#cv.destroyAllWindows()

#cv.imwrite('chart.png', img)

while True:
    bol, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.line(gray, (0,0), (100,100), (0,255,255), 20)
    cv.imshow('frame', gray)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
