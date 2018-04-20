# -*- coding: utf-8 -*-
#
# @method   : 生成验证码
# @Time     : 2018/4/20
# @Author   : wooght
# @File     : verification_code.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


def save_img():
    for _ in range(10000):
        img = Image.new(mode='RGB', size=(28, 28), color=(255, 255, 255))
        font = ImageFont.truetype("image/xdd.ttf", size=22)
        draw = ImageDraw.Draw(img, mode="RGB")
        # 画点
        for _ in range(50):
            fill = np.random.randint(0, 255, size=(1, 3))
            position = np.random.randint(0, 28, size=(1, 2)).tolist()
            draw.point(position[0], fill=tuple(fill[0]))

        # 画线
        position = np.random.randint(0, 28, size=(1, 4)).tolist()
        draw.line(position[0], fill=tuple(fill[0]))

        # 写字
        num = random.randint(0, 9)
        draw.text([5, 0], text=str(num), fill='red', font=font)
        with open('image/'+str(num)+'/'+str(random.randint(0, 99999999))+'.png', 'wb') as f:
            img.save(f, format='png')

save_img()


#
# img = Image.new(mode='RGB', size=(400,400), color=(255, 255, 255))
# with open("pic.png",'wb') as f:
#     img.save(f, format='png')
#
# # img.show()
#
# # 创建画笔
# draw1 = ImageDraw.Draw(img, mode="RGB")
#
# # 画点
# draw1.point([100, 100], fill="red")
# draw1.point([120, 120], fill=(0, 0, 0))
#
# # 画线
# draw1.line((50, 50, 200, 200), fill='red')
#
# # 画圆
# draw1.arc((100, 100, 400, 400),0, 360, fill='blue')
#
# # 写字
# font1 = ImageFont.truetype("image/xdd.ttf", size=120)
# draw1.text([0, 200], '123321', 'red', font=font1)
# out = img.rotate(20)  # 旋转
# out.show()
# print(img)