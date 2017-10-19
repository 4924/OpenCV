# Standard imports
import cv2 as cv
import numpy as np
import os
import math
import pickle
import time

import sys
import time
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print("Error: specify an IP to connect to!")
    exit(0)

ip = sys.argv[1]

NetworkTables.initialize(server=ip)

sd = NetworkTables.getTable("SmartDashboard")

cap = cv.VideoCapture("http://10.13.37.51/mjpg/video.mjpg")

while True:
    bol, frame = cap.read()

    im = frame

    #cv.imshow('color', im)
    im2 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #cv.imshow('gray', im2)

    im4 = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    im6 = cv.inRange(im4, np.array([54, 166, 144]), np.array([101, 220, 255]))
    #cv.imshow('inRange', im6)

    kerneld = np.ones(( 4, 4 ),np.uint8)
    im8 = cv.dilate(im6,kerneld,iterations = 1)
    #cv.imshow('big', im8)
    kernel = np.ones((9,2),np.uint8)
    im8 = cv.morphologyEx(im8, cv.MORPH_OPEN, kernel, iterations = 1)
    #cv.imshow('open', im8)


    imc, contours, hierarchy = cv.findContours(im8,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


    thing = []
    for c in contours:
        M = cv.moments(c)
        cX = int(M["m10"] / (M["m00"]+0.000000000001))
        cY = int(M["m01"] / (M["m00"]+0.000000000001))
        x,y,w,h = cv.boundingRect(c)
        aspect_ratio = float(w)/h
        if(aspect_ratio<.85):
            thing.append([cX,cY, c])

    def dist(first, second):
        return math.sqrt(abs(first[0]-second[0])**2 + abs(first[1]-second[1])**2)

    i = 0
    maxs = 9999
    maxarray = []
    while i < len(thing):
        for p in thing[i+1:]:
            if dist(thing[i], p) < maxs:
                maxs = dist(thing[i], p)
                maxarray = [thing[i][2], p[2]]
        i += 1


    im2 = cv.cvtColor(im2, cv.COLOR_GRAY2BGR)
    cv.drawContours(im2, maxarray, -1, (0,0,255), 1)
    done = []
    for m in maxarray:
        M = cv.moments(m)
        done.append(int(M["m10"] / (M["m00"]+0.000000000001)))
    if len(maxarray) > 0:
        if math.fabs(((sum(done)/len(done))/320)-.5) < .2:
            #print((((sum(done)/len(done))/320)-.5)*3)
            sd.putNumber('cameraX', -(((sum(done)/len(done))/320)-.5)*3)
        elif math.fabs(((sum(done)/len(done))/320)-.5) < .4:
            #print((30+((sum(done)/len(done))/320)-.5)*1.5)
            sd.putNumber('cameraX', -(((sum(done)/len(done))/320)-.5)*1.5)
        else:
            #print(50+((sum(done)/len(done))/320)-.5)
            sd.putNumber('cameraX', -(((sum(done)/len(done))/320)-.5))

    #cv.imshow('title', im2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
