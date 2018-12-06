#
# 生成不定长的验证码：字体、大小和长度可变，保持为文件！
#

from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z']
Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z']


##生成n位验证码 这里n=4
def random_captcha_text(char_set=number + alphabet + Alphabet, captcha_size=6):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


##使用ImageCaptcha库生成验证码
import os
def gen_captcha_text_and_image():
    # width = 160, height = 60, fonts = None, font_sizes = None
    # image = ImageCaptcha()
    # image = ImageCaptcha(width=300, height=200, fonts=['C:\Windows\Fonts\Calibri.ttf'], font_sizes=(42, 50, 56))
    image = ImageCaptcha(width=400, height=160, fonts=['C:\Windows\Fonts\Georgia.ttf'], font_sizes=(120, 120))
    # image = ImageCaptcha(width=300, height=200, fonts=['D:\Anaconda3\Lib\site-packages\captcha\data\DroidSansMono.ttf'], font_sizes=(100, 120, 120))

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)

    # 保存为文件

    return captcha_text, captcha_image


if __name__ == '__main__':
    ##展示验证码
    text, image = gen_captcha_text_and_image()
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()
