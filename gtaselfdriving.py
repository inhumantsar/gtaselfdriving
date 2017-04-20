from __future__ import print_function
import numpy as np
from PIL import ImageGrab as ig
import cv2
import time

BBOX=(0,40,1024,808)

def grabscreen():
    lt = time.time()
    while(True):
      numpied = np.array(ig.grab(bbox=BBOX), dtype='uint8')
      cv2.imshow('window', cv2.cvtColor(numpied, cv2.COLOR_BGR2RGB))
      ct = time.time()
      print(fps(lt, ct))
      lt = ct

      # bail on 'q' keypress
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

def fps(prev_time, cur_time):
    return round((1 / (cur_time - prev_time)), 2)

if __name__ == '__main__':
    grabscreen()
