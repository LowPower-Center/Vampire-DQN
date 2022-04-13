import ctypes
import time
import GameData as data
SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
SPACE=0x39
ESC=0x01

NP_2 = 0x50
NP_4 = 0x4B
NP_6 = 0x4D
NP_8 = 0x48

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functionsw

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))



delay=0.1

def up():
    PressKey(W)
    time.sleep(delay)
    ReleaseKey(W)


# 下
def left():
    PressKey(A)
    time.sleep(delay)
    ReleaseKey(A)



def down():
    PressKey(S)
    time.sleep(delay)
    ReleaseKey(S)



def right():
    PressKey(D)
    time.sleep(delay)
    ReleaseKey(D)

def pause(t=0.1):
    PressKey(ESC)
    time.sleep(t)
    ReleaseKey(ESC)
def confirm(t=0.1):
    PressKey(SPACE)
    time.sleep(t)
    ReleaseKey(SPACE)
def restart():
    for i in range(8):
        confirm(1)
def choose(num):
    if num ==1:
        confirm()
    elif num ==2:
        down()
        confirm()
    else:
        down()
        down()
        confirm()
if __name__ == '__main__':
    while True:
        HP=data.get_self_HP()
        print(HP)
        if HP<5:
            time.sleep(0.5)
            choose(3)
        if 5<HP<15:
            time.sleep(delay)
            HP=data.get_self_HP()
            if 5<HP<15:
                restart()
            else: pass
