from GameCapture import ROI , LEVEL_UP,LEVEL,TIME,grab_screen,HP,EXP
import cv2.cv2 as cv2
import numpy as np


def get_self_HP():
    img = grab_screen(region=HP)
    img=cv2.resize(img,(100,25))
    canny = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 0, 100)
    #    cv2.imshow("1",canny)
    #    cv2.waitKey(0)
    value = canny.argmax(axis=-1)
    return np.median(value)

def get_self_exp():

    img = grab_screen(region=EXP)

    img=cv2.resize(img,(100,25))
    canny = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 0, 100)
#    cv2.imshow("1",canny)
#    cv2.waitKey(0)
    value = canny.argmax(axis=-1)
    return np.median(value)
def get_self_lv():
    img = grab_screen(region=LEVEL)
    return
def is_level_up():
    img=grab_screen(region=LEVEL_UP)
    return



def get_items():
    pass


def get_state(state):
    img= ROI(state,*HP)
    img = cv2.resize(img, (100, 25))
    canny = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 0, 100)
    #    cv2.imshow("1",canny)
    #    cv2.waitKey(0)
    value1 = canny.argmax(axis=-1)
    img=ROI(state,*EXP)
    img = cv2.resize(img, (100, 25))
    canny = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 0, 100)
    #    cv2.imshow("1",canny)
    #    cv2.waitKey(0)
    value2 = canny.argmax(axis=-1)
    return np.median(value1),np.median(value2)

