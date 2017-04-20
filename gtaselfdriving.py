import numpy as np
from PIL import ImageGrab as ig
import cv2

BBOX = (0,40,1600,1240)

while(True):
  raw = ig.grab(bbox=BBOX)
  numpied = np.array(raw.getdata(), dtype='uint8').reshape((raw.size[1], raw.size[0], 3))
  cv2.imshow('window', numpied)
  
  # bail on 'q' keypress
  if cv2.waitKey(25) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    break
