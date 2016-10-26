# Standard imports
import cv2 as cv
import numpy as np
import os
import pickle
#424
#425
#495
#496
#66
#67
#68
#69
#70
#71
#72
#73
#74
#75
#76
#83
#91
#92
array = []
# Read image
locate = 0
for s in os.listdir("files"):
    im = cv.imread("files/"+s, cv.IMREAD_COLOR)

    #print s

    #def callback(x):
    #    pass


    im = cv.resize(im, (320,240))
    #im = cv.bitwise_not(im)

    #im2 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #cv.imshow('gray', im2)

    #th, im3 = cv.threshold(im2, 50, 255, cv.THRESH_TOZERO);
    im4 = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    im6 = cv.inRange(im4, np.array([79, 162, 50]), np.array([98, 255, 255]))
    ##im6 = cv.cvtColor(im5, cv.COLOR_HSV2BGR)
    #cv.imshow('threshold', im3)
    #cv.imshow('color', im)
    #cv.imshow('inRange', im6)


    contours, hierarchy = cv.findContours(im6,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


    lan = [0]*5
    thing = [0]*5
    for c in contours:
        areaConvex = cv.contourArea(cv.convexHull(c))
        perim = cv.arcLength(c,True)
        for i in range(0,3):
            if( areaConvex + perim >=lan[i]):
                lan.insert(0, areaConvex + perim)
                thing.insert(0, c)
                break


#(x,y),(MA,ma),angle = cv.fitEllipse(thing)
#print angle
#for i in range(0, 3):
#    x,y,w,h = cv.boundingRect(thing[0])
#    crop_im = im[y:y+h, x:x+w]
#    cv.imwrite("file_output_new/"+str(i)+"|"+s, crop_im)
#cv.imshow('crop', crop_im)


#save = thing
#saveHandler = open('example.contour', 'w')
#pickle.dump(save, saveHandler)

#saveHandler = open('example.contour', 'r')
#savedThing = pickle.load(saveHandler)
#print cv.matchShapes(thing, savedThing, 1, 0.0)

#print cv.contourArea(cv.convexHull(thing))
#print cv.contourArea(cv.convexHull(thing))/cv.contourArea(thing)*50

    im4 = cv.cvtColor(im6, cv.COLOR_GRAY2BGR)
    cv.drawContours(im4, thing[0:3], -1, (0,255,0), 1)
    cv.imwrite("filesas/"+s, im4)
#cv.imshow('title', im4)
#cv.waitKey(0)
#cv.destroyAllWindows()
#
#cv.namedWindow('thresh')
#
## Track bars
#cv.createTrackbar('lH','thresh', 0, 180, callback)
#cv.createTrackbar('lS', 'thresh', 0, 255, callback)
#cv.createTrackbar('lV', 'thresh', 0, 255, callback)
#cv.createTrackbar('uH', 'thresh', 0, 180, callback)
#cv.createTrackbar('uS', 'thresh', 0, 255, callback)
#cv.createTrackbar('uV', 'thresh', 0, 255, callback)
#
## capture video
#
#
#while True:
#    # Getting values from track bars
#    lH = cv.getTrackbarPos('lH', 'thresh')
#    uH = cv.getTrackbarPos('uH', 'thresh')
#    lS = cv.getTrackbarPos('lS', 'thresh')
#    uS = cv.getTrackbarPos('uS', 'thresh')
#    lV = cv.getTrackbarPos('lV', 'thresh')
#    uV = cv.getTrackbarPos('uV', 'thresh')
#
#
#    lowerb = np.array([lH, lS, lV], np.uint8)
#    upperb = np.array([uH, uS, uV], np.uint8)
#
#    #print lowerb
#    #print upperb
#
#    frame = cv.inRange(im4, lowerb, upperb)
#    cv.imshow('thresh', frame)
#
#    if cv.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap.release()
#cv.destroyAllWindows()
#
