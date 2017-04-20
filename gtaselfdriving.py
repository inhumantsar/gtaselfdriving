from __future__ import print_function

import numpy as np
import cv2
import time

from PIL import ImageGrab as ig
from directkeys import PressKey, W, A, S, D

# starts in the top left corner, 40px down to cut off the titlebar
BBOX=(0,40,1024,793)
EDGE_T=(250,350)
ROI_V=np.array([[10,660],[10,400],[385,265],[640,265],[1024,400],[1024,793]], np.int32)


def process(orig):
    # convert to greyscale, edge detect
    new = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY),
            threshold1=EDGE_T[0], threshold2=EDGE_T[1])

    # trim to roi
    new = roi(new)

    # to make edge detection a b it chiller
    new = cv2.GaussianBlur(new, (5,5), 0)

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    lines = cv2.HoughLinesP(new, 1, np.pi/180, 180, 20, 15)
    draw_lines(new, lines)
    return new

def roi(orig, vertices=ROI_V):
    # create a mask for the ROI vertices
    mask = np.zeros_like(orig)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(orig, mask)

def draw_lines(orig, lines):
    if not isinstance(lines, list):
        print("lines is not a list")
        return

    if len(lines) < 1:
        print("no lines present")
        return

    for line in lines:
        coords = line[0]
        cv2.line(orig, (coords[0], coords[1]), (coords[2], coords[3]), (255,255,255), 4)

def grabscreen():
    lt = time.time()
    while(True):
        # grab a screencap of everything inside the area defined by bbox
        numpied = np.array(ig.grab(bbox=BBOX), dtype='uint8')

        # display the screencap, correcting the colour as we go.
        cv2.imshow('window', process(numpied))
        # cv2.imshow('window', cv2.cvtColor(numpied, cv2.COLOR_BGR2RGB))

        # measure everything
        ct = time.time()
        print("{} fps".format((1 / (ct - lt))))
        lt = ct

        # bail on 'q' keypress
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    grabscreen()
