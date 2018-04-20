# -*- coding: utf-8 -*-
#
# @method   : 测试数据-工厂
# @Time     : 2018/3/30
# @Author   : wooght
# @File     : test_data.py

import numpy as np


# 性别样本
def gender_sample():
    # 男 H:160-190, T:110-210, X:39-44
    # 女 H:150-175, T:80-160, X:36-41
    man_h = np.random.randint(160, 190, size=(100, 1))
    man_t = np.random.randint(100, 210, size=(100, 1))
    man_x = np.random.randint(39, 44, size=(100, 1))
    man = np.column_stack([man_h, man_t, man_x])
    # for _ in 的应用,循环生成指定格式数组的方式
    wman = [[np.random.randint(150, 175), np.random.randint(80, 160), np.random.randint(36, 41)] for _ in range(100)]
    person_train = np.row_stack([man, wman])
    category_train = np.random.randint(1, size=(200))
    category_train[-100:] = 1
    return person_train, category_train


def float_rand(arr, x):
    return np.array([j + np.random.rand()*x for j in arr])


def float_shows(t, w, a):
    sample_arr = [0.1, 0.12, 0.13]
    return t*np.random.choice(sample_arr)+\
           w*np.random.choice(sample_arr)+\
           a*np.random.choice(sample_arr)


# 广告效果样本
def ad_sample():
    # 自变量: TV,WEB,APP  线性数据
    # 因变量: shows
    tv = float_rand(np.linspace(1, 20, 200), 0.2)
    web = float_rand(np.linspace(1, 50, 200), 0.3)
    app = float_rand(np.linspace(1, 100, 200), 0.5)
    shows = np.array([float_shows(x, y, z) for x,y,z in zip(tv, web, app)])
    result = np.column_stack([tv, web, app, shows])
    return result


def f(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 0.1 * x1 + 3
    return y


# 常用等差队列正弦样本
def sin_data():
    # 训练数据500个,测试数据100个 训练数据有-0.5~0.5的噪声
    x1_train = np.linspace(0, 50, 500)  #等差数列 0,50 共500个
    x2_train = np.linspace(-10, 10, 500)
    data_train = np.array([[x1, x2, f(x1, x2) + (np.random.random(1) - 0.5)] for x1, x2 in zip(x1_train, x2_train)])
    x1_test = np.linspace(0, 50, 100) + 0.5 * np.random.random(100)
    x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
    data_test = np.array([[x1, x2, f(x1, x2)] for x1, x2 in zip(x1_test, x2_test)])
    return data_train, data_test
