##--계단함수구현하기
import numpy as np
import matplotlib.pylab as plt

#계단 함수 정의
def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.arange(-5.0, 5.0, 0.1)  #-5에서 5 사이 0.1간격의 넘파이 배열
y = step_function(x)

#그래프
plt.plot(x,y)
plt.ylim(-0.1, 1.1)  #y축의 범위 지정
plt.show()

##--시그모이드함수구현
#시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

#그래프
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show

##--relu함수구현
def relu(x):
    return np.maximum(0,x)

##--다차원배열
import numpy as np

#1차원
A = np.array([1,2,3,4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

#2차원 (행렬)
B = np.array([[1,2], [3,4], [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)

#행렬의 곱
A = np.array([[1,2], [3,4]])
B = np.array([[5,6],[7,8]])

np.dot(A,B)

#형상이 다른 행렬의 곱
A = np.array([[1,2,3], [4,5,6]])
B = np.array([[1,2],[3,4], [5,6]])

np.dot(A,B)

##--신경망행렬곱
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])

Y = np.dot(X,W)
print(Y)

##--3층신경망 구현
def init_network():
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  #1층 가중치
        network['b1'] = np.array([0.1, 0.2, 0.3])  #1층 편향
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  #2층 가중치
        network['b2'] = np.array([0.1,0.2])  #2층 편향
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])  #3층 가중치
        network['b3'] = np.array([0.1, 0.2])  #3층 편향

        return network

def forward(network, x):
    W1, W2, W3 = network["W1"], network['W2'], network['W3']
    b1, b2, b3 = network["b1"], network['b2'], network['b3']

    a1 = np.dot(x, W1) +b1 #입력값*가중치 +편향
    z1 = sigmoid(a1)  #1층 활성화함수 출력값
    a2 = np.dot(z1, W2) +b2  #2층
    z2 = sigmoid(a2)  #2층 활성화함수 출력값
    a3 = np.dot(z2, W3) +b3  #3층
    y = a3  #출력층 활성화 함수로 항등함수 사용

    return y

network = init_network()
x = np.array([1.0, 0.5])  #입력값
y = forward(network, x)
print(y)

##--소프트맥스 함수 구현
def softmax(a):
        c = np.max(a)
        exp_a = np.exp(a-c)  #오버플로 대책
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
np.sum(y)

##--손글씨 숫자 인식
from google.colab import drive
drive.mount('/content/drive')

import sys, os
sys.path.append("/content/drive/MyDrive/Colab Notebooks/밑바닥딥러닝/dataset")
from mnist import load_mnist  ##mnist.py파일에 정의된 load_mnist()함수 이용
from PIL import Image

#다운로드
(X_train, t_train), (X_test, t_test) = \  #(훈련이미지, 훈련레이블), (시험이미지, 시험레이블)
  load_mnist(flatten=True, normalize=False)
  
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
  
img = X_train[0]  #첫번째 훈련 이미지
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)

##--MNIST 신경망
import numpy as np
import pickle  #프로그램 실행 중에 특정 객체를 파일로 저장, 데이터 빠르게 준비 가능

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("/content/drive/MyDrive/Colab Notebooks/밑바닥딥러닝/dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f) #학습된 가중치 매개변수: 가중치, 편향 매개변수가 딕션너리 변수로 되어 있음

    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network['W2'], network['W3']
    b1, b2, b3 = network["b1"], network['b2'], network['b3']

    a1 = np.dot(x, W1) +b1 #입력값*가중치 +편향
    z1 = sigmoid(a1)  #1층 활성화함수 출력값
    a2 = np.dot(z1, W2) +b2  #2층
    z2 = sigmoid(a2)  #2층 활성화함수 출력값
    a3 = np.dot(z2, W3) +b3  #3층
    y = softmax(a3)

    return y

#정확도 평가
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  #확률이 가장 높은 원소의 인덱스
    if p == t[i]:
        accuracy_cnt +=1

print("Accuracy:" +str(float(accuracy_cnt / len(x))))  #정확도 : 맞힌 횟수 / 전체 이미지 숫자

#배치 처리
x, t = get_data()
network = init_network()

batch_size = 100  #배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  #axis=1 : 행을 따라 최대값, axis=0 :열을 따라 최대값
    accuracy_cnt +=np.sum(p==t[i:i+batch_size])  #True인, p와 t가 같은 개수 셈

print("Accuray:" + str(float(accuracy_cnt)/len(x)))