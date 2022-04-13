import cv2.cv2 as cv2
import numpy as np
import time
from GameCapture import grab_screen,LEVEL,Full_Screen,TIME,LEVEL_UP,Treasure

img=grab_screen(region=Treasure)
#cv2.imwrite("sample_{}.png".format(time.time()),img)
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", img)
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img)
    if cv2.waitKey(0)&0xFF==27:
        break
cv2.destroyAllWindows()