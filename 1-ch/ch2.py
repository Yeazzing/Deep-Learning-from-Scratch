#가중치와 편향 구현하기
import numpy as np

#AND 게이트 구현
def AND(x1, x2):
    x = np.array([x1,x2])  #입력값
    w = np.array([0.5, 0.5])  #가중치
    b = -0.7  #편향
    tmp = np.sum(w*x) +b
    if tmp <=0:
        return 0
    else:
        return 1

#NAND 게이트 구현
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  #AND와는 가중치만 다름
    b = 0.7
    tmp = np.sum(w*x) +b
    if tmp <=0:
        return 0
    else:
        return 1

#OR 게이트 구현
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])  #AND와는 가중치만 다름
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0 :
        return 0
    else:
        return 1

def XOR (x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR (x1, x2)
    y = AND(s1, s2)
    return y