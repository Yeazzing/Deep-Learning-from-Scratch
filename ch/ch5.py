# -*- coding: utf-8 -*-
"""ch5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EORlr2h_eojASwMv1UYIvc5ErlB5NtMX
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import sys, os
# %cd /content/drive/MyDrive/Colab Notebooks/밑바닥딥러닝
from common.functions import *
from common.gradient import numerical_gradient
from collections import OrderedDict

import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

"""단순한 계층 구현하기"""

#곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self, dout):
        dx = dout*self.y  #상류에서 넘어온 미분(dout)에 순전파 때의 값을 서로 바꿔 곱함
        dy = dout*self.x

        return dx, dy

#사과 쇼핑 예 적용
apple = 100
apple_num = 2
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

#역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)

#덧셈 계층
class AddLayer:
    def __init__(self):
        pass  #초기화 필요 x

    def forward(self, x, y):
        out = x+y
        return out

    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy

#사과와 귤 쇼핑 예 적용
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()

#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)

#역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dorange, dorange_num, dtax)

"""활성화 함수 계층 구현하기"""

#relu 계층
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)  #True/False로 구성된 넘파이 배열, 0이하면 True
        out = x.copy()
        out[self.mask] = 0 #x가 0이하이면(True면) 0

    def backward(self, dout):
        dout[self.mask] = 0  #순전파 때 x가 0이하이면 역전파 0
        dx = dout

        return dx

#sigmoid 계층
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out

    def backward(self, dout):
        dx = dout * (1.0-self.out)*self.out

        return dx

"""Affine/Softmax 계층 구현하기"""

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x,self.W) +self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  #dout*다른입력값W의 전치행렬 (대응하는 차원의 원소 수가 일치하도록 하기 위해 전치)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)  #각 데이터에 대한 미분을 데이터마다 더해서 구해야 함, 데이터 단위의 0번째 축에 대한 총합

        return dx

#softmaxwithloss
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  #손실
        self.y = None  #softmax의 출력
        self.t = None  #정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y -self.t) / batch_size

        return dx

"""오차역전파법 구현"""

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params = {}  #신경망의 매개변수 보관하는 딕셔너리 변수
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)  #정규분포를 따르는 난수로 초기화
        self.params['b1'] = np.zeros(hidden_size)  #0으로 초기화
        self.params['W2'] =  weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #계층 생성
        self.layers = OrderedDict()  #순서가 있는 딕셔너리{Affine1, Relu1, Affine2}
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()  #신경망의 마지막 계층

    def predict(self, x):  #예측 수행
        for layer in self.layers.values():  #{Affine1, Relu1, Affine2}순서대로 처리
            x = layer.forward()

        return x

    # x:입력 데이터, t: 정답 레이블
    def loss(self, x, t):  #손실 함수 값
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):  #정확도
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):  #수치미분방식으로 기울기 구하기
        loss_W = lambda W: self.loss(x,t)

        grads = {}  #기울기 보관하는 딕셔너리 변수
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):  #오차역전파법으로 기울기 구하기
        #순전파
        self.loss(x, t)

        #역전파
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layer.values())
        layers.reverse()  #{Affine1, Relu1, Affine2}역순으로
        for layer in layers:
            dout = layer.backward(dout)

        #결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].dW
        return grads

#기울기 검증
#데이터 읽기
(X_train, t_train), (X_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = X_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key +":" +str(diff))

#오차역전파법을 사용한 학습 구현 - ch4와 동일
(X_train, t_train), (X_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

#하이퍼파라미터
iters_num = 10000  #반복 횟수
train_size = X_train.shape[0]
batch_size = 100  #미니배치 크기
learning_rate = 0.1

network = TwoLayerNet(input_size= 784, hidden_size = 50, output_size= 10)

#1에폭당 반복 수
iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    #미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)  #60000개의 훈련 데이터에서 임의로 100개의 데이터 추리기
    x_batch = X_train[batch_mask]
    t_batch = t_train[batch_mask]

    #기울기 계산
    grad = network.gradient(x_batch, t_batch)  #성능 개선판 : 오차역전파법
    #grad = network.numerical_gradient(x_batch, t_batch)

    #매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2' ):
        network.params[key] -= learning_rate *grad[key]

    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #1에폭당 정확도 계산
    if i%iter_per_epoch ==0:
        train_acc = network.accuracy(X_train, t_train)
        test_acc = network.accuracy(X_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc : " + str(train_acc) +"," +str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()