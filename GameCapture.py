# Done by Frannecklp

import cv2.cv2 as cv2
import numpy as np
import win32gui, win32ui, win32con, win32api


Full_Screen=(0, 226, 959, 825)
HP=(460, 534, 502, 538)
TIME=(445,265,513,283)
EXP=(10,234,960,248)
LEVEL=(930,230,960,250)
LEVEL_UP=(307,295,655,764)
Treasure=(308,297,652,762)
State=()




def ROI(img,x,y,x_w,y_h):
    if x>x_w:
        x,x_w=x_w,x
    if y>y_h:
        y,y_h=y_h,y
    return  img[y:y_h+1,x:x_w+1]


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left +1
        height = y2 - top +1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)