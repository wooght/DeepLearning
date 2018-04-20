# -*- coding: utf-8 -*-
#
# @method   : 读取图片
# @Time     : 2018/4/20
# @Author   : wooght
# @File     : load_data.py

import os
from PIL import Image
import numpy as np


def load_data(path):
    img_data = np.empty((10000, 1, 28, 28), dtype='float32')
    target = np.empty((10000,), dtype='int32')
    now_i = 0
    for i in range(10):
        dir = path+'/'+str(i)
        imgs = os.listdir(dir)  # 读取目录所有文件名
        nums = len(imgs)
        for j in range(nums):
            img = Image.open(dir + '/' + str(imgs[j])).convert('L')  # 将RGB转换为灰度图像
            # img.show()
            img_data[now_i, :, :, :] = img  # 广播赋值
            target[now_i] = i
            now_i += 1
    permutation = np.random.permutation(img_data.shape[0])  # 获取打乱后的行
    img_data = img_data[permutation, :, :, :]
    target = target.reshape((10000,1))
    target = target[permutation, :]
    return img_data, target


if __name__ == '__main__':
    img, target = load_data('.')
    print(img[9995])