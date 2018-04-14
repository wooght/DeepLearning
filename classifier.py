# -*- coding: utf-8 -*-
#
# @method   : keras 简单分类
# @Time     : 2018/4/14
# @Author   : wooght
# @File     : classifier.py
# 相关词条
# backend 后台,后端

import numpy as np
np.random.seed(256)  # 随机数据再现
from keras.datasets import mnist  # mnist keras自带mnist数据
from keras.utils import np_utils  # keras对np进行改进的工具
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 数据说明
# 共60000张图片,是手写的0-9的图片,图片大小28*28像素
# 每个图片是一个向量,shape:28*28,及784个特征, 每个特征是表示颜色值(0-255,RGB)

# 数据预处理(preprocessing)
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # 数据预处理,标准化  将0-255的值转换成0-1
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # 数据预处理
y_train = np_utils.to_categorical(y_train, num_classes=10)  # 分类向量转换,10类
actual_y_test= y_test
y_test = np_utils.to_categorical(y_test, num_classes=10)
# np_utils.to_categorical() 对数据进行绝对向量转换 如数据[5]会转换为向量[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# 及相应位置为1 其余位置为0

print(X_train[0], X_train[0].shape)

# 建立model, Sequential一层一层递进的神经模型
model = Sequential([
    Dense(32, input_dim=784),  # 只有第一层要传递 input_dim 28*28 = 784, outinput_dim输出32个Features
    Activation('relu'),  # 激活函数[非线性函数(及不是直线,需要积分和微分实现曲线)用到激励函数], 用于隐层神经元输出(relu,Sigmoid)
    # 层少的话,relu,Sigmoid都可以选择,层多的话,relu多用在卷积网络,tenh用在循环神经网络
    Dense(10),  # 第二层,将特征降低为10个特征,这里0-9共10个类
    Activation('softmax'),  # Softmax 用于多分类神经网络输出, Linear用于回归神经网络输出（或二分类问题)
])

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  # RMSProp优化函数  优化函数??

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',  # categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
              metrics=['accuracy'])  # 评估函数 accuracy 正确预测的比例

model.fit(X_train, y_train, epochs=3, batch_size=32)
loss, accuracy = model.evaluate(X_test, y_test)  # 评估
print(loss, accuracy)

# 和实际对比查看
predict = model.predict(X_test)
result = []
for i in predict:
    the_nums = list(np.where(i==i.max()))[0][0]  # np.where(arr=num) 返回arr中num的索引位置
    result.append(the_nums)
test_predict = np.array(result)
db_test = np.column_stack([result, actual_y_test])
print(db_test)