# -*- coding: utf-8 -*-
#
# @method   : keras 线性回归
# @Time     : 2018/4/12
# @Author   : wooght
# @File     : keras_test.py
# 相关词条:
# Sequential [sɪˈkwɛnʃəl] 相继的,按次序的 序贯模型
# batch [bætʃ] 一批,批次
# layers [leɪəs] 层,层次
# compile [kəmˈpaɪl] 编译
# optimizer [ɑ:ptɪmaɪzər] 优化器
# tensor 张量,用张量标识广泛的数据类型,0阶张量就是一个数,1阶张量就是向量...

import numpy as np
np.random.seed(2128)  # 设定随机整数值,以便以后运行随机数相同,及再现
from keras.models import Sequential
from keras.layers import Dense  # 全链接层
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # 打乱数据
Y = 0.5 * X + 2 + np.random.normal(loc=0, scale=0.05, size=(200, ))  # normal 正太分布,loc中心点,scale标准差
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# 建立模型
model = Sequential()
# Dense层,指全链接层 input_dim,output_dim输入输出维度 input_shape 输入shape
model.add(Dense(units=1, input_dim=1))  # units 指输出维度
# 模型只有一个层,切没有激活函数,则默认线性激活函数

# 编译,激活模型compile共三个参数: loss误差函数'mse'均方误差 , optimizer优化器sgd梯度下降法, metrics 评估模型的指标
model.compile(loss='mse', optimizer='sgd')

# 训练
# for step in range(301):
#     cost = model.train_on_batch(X_train, Y_train)  # 一批一批的训练, 返回compile时,指定loss函数返回值(误差值)
#     if step % 100 == 0:
#         print('train cost: ', cost)
model.fit(X_train, Y_train, verbose=1, epochs=100, batch_size=20)
# batch_size: 指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
# verbose:训练时显示实时信息，0表示不显示数据，1表示显示进度条，2表示用只显示一个数据
# epochs: 指被训练轮数

# 评估,检查
score = model.evaluate(X_test, Y_test, batch_size=40)  # 返回loss函数返回的值 batch_size 指定批次大小
print('test cost:', score)
# layers 层,我们要获取第一层(这里只有一层)网络的返回值 get_weights()
W, b = model.layers[0].get_weights()  # 线性回归方程 y=wx+b  w指斜率,b指截距
print('Weights=', W, '\nbiases=', b)

# 预测
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()