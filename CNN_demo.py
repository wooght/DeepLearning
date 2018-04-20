# -*- coding: utf-8 -*-
#
# @method   : CNN 图像分类Demo
# @Time     : 2018/4/20
# @Author   : wooght
# @File     : CNN_demo.py

from wooght.image.load_data import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
img_data, target= load_data('wooght/image')
img_data = img_data/255  # 规范数据
target = np_utils.to_categorical(target, num_classes=10)
print(target)

model  = Sequential()  # 全连接层

model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first'
))

model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first'
))
model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer=Adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(img_data[:8000,:], target[:8000,:], epochs=1, batch_size=32)

loss, accuracy = model.evaluate(img_data[8000:,:], target[8000:,:])
print('loss:',loss,'\naccuracy:', accuracy)


