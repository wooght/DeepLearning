# -*- coding: utf-8 -*-
#
# @method   : 感知机
# @Time     : 2018/4/17
# @Author   : wooght
# @File     : perceptron.py

import numpy as np
import matplotlib.pyplot as plt

# 二元神经网络--感知机
# 数据样本 y=x+random 及x+random-y>0 为1类,<0为-1类
np.random.seed(1123)
data_one = np.array([[x, x + np.random.randint(4, 10)] for x in range(50)])
data_two = np.array([[x, x - np.random.randint(4, 10)] for x in range(50)])
data = np.row_stack([data_one, data_two])
target = np.ones((100, 1))
target[50:, 0] = -1
target = target.reshape(100)


def matplot(w1, w2, b):
    plt.plot(data_one[:, 0], data_one[:, 1], 'r.')
    plt.plot([0, 50], [0, 50], 'b-', linewidth=1)
    plt.plot(data_two[:, 0], data_two[:, 1], 'k.')
    predict = np.array([[x, -(w1 * x + b) / w2] for x in range(50)])
    plt.plot(predict[:, 0], predict[:, 1], 'k-')
    plt.show()


# 根据w和b 前向验证target,如果错误,调整w和b
w1 = 1
w2 = 1
b = 0
# for _ in range(500):
#     for i in range(100):
#         s = w1 * data[i][0] + b + w2 * data[i][1]
#         if (s <= 0 and target[i] > 0) or (s > 0 and target[i] < 0):
#             w1 += data[i][0] * target[i]  # 梯度下降法,向目标缩进
#             w2 += data[i][1] * target[i]  # 为什么w1,w2同时要修改, 因为在y=ax+b 中 a可能是分数, w1/w2才有可能得到分数
#             b += target[i]
#             matplot(w1, w2, b)
#
# print(w1, w2, b)


# 当不是线性的时候,用到激励函数 就变成了神经网络

# 非线性激励函数(sigmoid)
# sigmoid(x) = 1/(1+e**(-x))
# deriv=ture 是求的是导数
def nonlin(x, deriv=False):
    if deriv == True:
        return 1.0 - np.tanh(x) * np.tanh(x)
    return  np.tanh(x)

from decimal import Decimal
from math import floor
# input dataset
X_one = np.array([[x, floor(x**2/30)+np.random.randint(2,20)] for x in np.arange(1, Decimal(51))])
X_two = np.array([[x, floor(x**(5/3)/30)+np.random.randint(2,20)] for x in np.arange(1, Decimal(51))])
X = np.row_stack([X_one, X_two])
y = np.ones((100,1))
y[50:] = -1
np.random.seed(1)
plt.plot(X_one[:,0], X_one[:,1], 'r.')
plt.plot(X_two[:,0], X_two[:,1], 'b.')
# plt.show()

# 神经网络 默认weights
syn0 = 2 * np.random.random((2, 1)) - 1
l1 = object  # 输出层
l1_delta = object  # 隐藏层权值增量

# 迭代次数
for iter in range(3000):
    # 前向传播
    # l0也就是输入层
    l0 = X
    # l1是隐藏层
    l1 = nonlin(np.dot(l0, syn0))  # np.dot矩阵内积 通过激励函数非线性转换,得到隐藏层结果

    # 误差
    l1_error = y - l1  # 分类误差,如果分类全部正确,则得到0损失
    if iter % 100 == 0:
        print(np.mean(np.abs(l1_error)))


    l1_delta = l1_error * nonlin(l1, True)  # 误差 * 分类结果激励函数导数 = 权重增量(损失量)

    # 修改权值
    syn0 += np.dot(l0.T, l1_delta)  #  向目标值缩进/接近

print("Output After Training:")
print(l1)
print(l1_delta)
print(syn0)  # 输出层 ,一个训练好的神经网络(特征权值)
# 验证
print(nonlin(np.dot(np.array([[1, 2], [5, 6], [50, 39]]), syn0)))
predict = np.array([[x, -(syn0[0] * x + 0) / syn0[1]] for x in range(1, 51)])
print(predict)
plt.plot(predict[:,0], predict[:,1], 'y-')
plt.show()
