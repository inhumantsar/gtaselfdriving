from __future__ import print_function

import numpy as np
import cv2
import time

from PIL import ImageGrab as ig
from directkeys import PressKey, W, A, S, D

#               [385,265] ---- [640,265]
#           /                            \
# [10,400]                                  [1024, 400]
#     |                                         |
# [10,660]                                  [1024, 793]
ROI_V=np.array([[10,793],[10,400],[385,265],[640,265],[1024,400],[1024,793], [950,793], [600,350], [400,350], [100,793]], np.int32)
BBOX=(0,40,1024,793) # starts in the top left corner, 40px down to cut off the titlebar
EDGE_T=(250,350)

g_slopes = {'p': [], 'n': []}
g_intercepts = {'p': [], 'n': []}
min_top_ys = []

def process(orig):
    # convert to greyscale, edge detect
    # new = cv2.Canny(cv2.cvtColor(yellow_to_white(orig), cv2.COLOR_BGR2GRAY),
    #         threshold1=EDGE_T[0], threshold2=EDGE_T[1])
    new = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY),
            threshold1=EDGE_T[0], threshold2=EDGE_T[1])

    # trim to roi. opencv is not happy if i try to do this first...?
    new = roi(new)

    # to make edge detection a bit chiller
    new = cv2.GaussianBlur(new, (5,5), 0)

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    lines = cv2.HoughLinesP(new, 1, np.pi/180, 180,      20,       15)
    try:
        l1, l2 = draw_lanes(orig,lines)
        cv2.line(orig, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(orig, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(new, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return new, orig

def get_line_info(lines):
    # each line contains two points, we need to get collect all of those individual points
    # into their own arrays, separated by axis
    x = np.reshape(lines[:, [0,2]], (1, len(lines) * 2))[0] # get x vals
    y = np.reshape(lines[:, [1,3]], (1, len(lines) * 2))[0] # get y vals

    # not 100% sure what's goin on with this line
    A = np.stack([x, np.ones(len(x))], axis=0)
    m, x = np.linalg.lstsq(A, y)[0]
    x = np.array(x)
    y = np.array(x * m + c).astype('int')
    return x, y, m, c


### an attempt at making draw_lanes less gross. largely from
### http://shrikar.com/self-driving-car-lane-detection/
# def draw_lanes(img, lines, color=[0,255,255], thickness=3):
#     global g_slopes
#     global g_intercepts
#     global min_top_ys
#
#     slopes = np.apply_along_axis(lambda x: (x[3] - x[1])/(x[2] - x[0]), 2, lines)
#     p_slopes = slopes > 0.5
#     p_lines = lines[p_slopes]
#     n_slopes = slopes < -0.5
#     n_lines = lines[n_slopes]
#
#     # use a rolling average slope+intercept
#     px, py, pm, pc = get_line_info(p_lines)
#     g_slopes['p'] = np.append(g_slopes['p'], [pm])
#     g_intercepts['p'] = np.append(g_intercepts['p'], [pc])
#     pm = g_slopes['p'][-20:].mean()
#     pc = g_intercepts['n'][-20:].mean()
#
#     nx, ny, nm, nc = get_line_info(n_lines)
#     g_slopes['n'] = np.append(g_slopes['n'], [nm])
#     g_intercepts['n'] = np.append(g_intercepts['n'], [nc])
#     nm = g_slopes['n'][-20:].mean()
#     nc = g_intercepts['n'][-20:].mean()
#
#     bot_left_y = img.shape[0]
#     bot_right_y = img.shape[0]
#     bot_left_x = int((bot_left_y - nc))
#     bot_right_x = int((bot_right_y + nc))
#
#     min_top_ys = np.append(min_top_ys, np.min([ny.min(), py.min()]))
#     avg_top_y = int(min_top_ys[-20:].mean())
#     top_left_y = avg_top_y
#     top_right_y = avg_top_y
#     top_left_x = int((top_left_y - nc) / nm)
#     top_right_x = int((top_right_y - nc) / nm)
#
#     draw_lines(img, [(top_left_x, top_left_y, bot_left_x, bot_left_y), (top_right_x, top_right_y, bot_right_x, bot_left_y)])


def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = img.shape[0]
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m, b = np.linalg.lstsq(A, y_coords)[0]

                if m == 0:
                    print('slope appears to be zero, skipping')
                    continue

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]

            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        print(str(e))


def roi(orig, vertices=ROI_V):
    # create a mask for the ROI vertices
    mask = np.zeros_like(orig)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(orig, mask)

def draw_lines(orig, lines):
    if not isinstance(lines, np.ndarray):
        print("lines is not a list, instead: {}".format(type(lines)))
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
        # cv2.imshow('window', process(numpied)[0])
        cv2.imshow('window', cv2.cvtColor(process(numpied)[1], cv2.COLOR_BGR2RGB))

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
