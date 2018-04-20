# -*- coding: utf-8 -*-
#
# @method   : keras CNN(卷神经)
# @Time     : 2018/4/16
# @Author   : wooght
# @File     : keras_CNN.py

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from PIL import Image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_data = X_train.reshape(-1,1,28,28)
print(img_data[0][0])
print(y_train)
img = Image.fromarray(img_data[1][0])
img.show()
# 数据预处理 preprocession
X_train = X_train.reshape(-1, 1, 28, 28)/255  # 三维变四维,(1, 28, 28) 1指图片高度,因为是黑白,所以只有1层
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)



model = Sequential()

# 添加第一层卷积
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=12,  # 卷积核的数目,及输出的维度
    kernel_size=6,  # 卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。 这里及5*5
    strides=1,  # 跨1个单位的步长(卷积步长)
    padding='same',     # Padding 抽离信息(边界处理),same指和以前一样 valid和之前较小(不包括边界)
    data_format='channels_first',  # data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置
))
# 激励函数 relu  卷神经最佳relu
model.add(Activation('relu'))

# 池化 最大值池化 在pool_size范围内,提取最大值
model.add(MaxPooling2D(
    pool_size=2,  # 整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子 2及(2, 2),将原来的[[1, 1],[1, 1]]变为[1, 1] 及取样一半
    strides=2,  # 步长 同Convolution2d
    padding='same',    # padding 同上
    data_format='channels_first',
))

# 添加第二层卷积  output shape: (?, 64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# 池化pool output shipe: (?, 64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train[:10000,:], y_train[:10000,:], epochs=1, batch_size=64,)

print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
W, b = model.layers[0].get_weights()
print(W.shape,b)
print(W)
W,b = model.layers[3].get_weights()
print(W.shape)
