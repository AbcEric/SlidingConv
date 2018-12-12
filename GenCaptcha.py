#
# 生成不定长的验证码：字体、大小、颜色和长度可变，保持为文件！长度为2~10，
#
# 1. 今后可通过二值化图片，判断是黑底白字还是白底黑字，进行统一；
# 2. 图片保存在training_data/captcha_img下；
# 3. 修改了image的generate_image(): 可根据情况修改颜色，是否添加干扰点和干扰线；
#
# Q: 宽度不够，感觉较为瘦高？小写英文字母通常瘦高，宽度为高度的一半？

from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
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

def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


def my_generate_image(image, chars, format='png'):
    background = random_color(238, 255)
    color = random_color(10, 200, random.randint(220, 255))

    # if random.randint(0, 1) == 0:
    #     background = random_color(238, 255)
    #     color = random_color(10, 200, random.randint(220, 255))
    # else:
    #     background = random_color(10, 30)
    #     color = random_color(200, 200, random.randint(220, 255))

    im = image.create_captcha_image(chars, color, background)
    image.create_noise_dots(im, color)
    image.create_noise_curve(im, color)
    im = im.filter(ImageFilter.SMOOTH)
    out = BytesIO()
    im.save(out, format=format)
    out.seek(0)
    return out

##使用ImageCaptcha库生成验证码
import os
def gen_captcha_text_and_image(textlen, fontsize, fontname):
    # width = 160, height = 60, fonts = None, font_sizes = None
    # image = ImageCaptcha()
    # image = ImageCaptcha(width=300, height=200, fonts=['C:\Windows\Fonts\Calibri.ttf'], font_sizes=(42, 50, 56))
    image = ImageCaptcha(width=int(fontsize*(textlen+1)/2), height=int(fontsize), fonts=[fontname], font_sizes=(fontsize-5, fontsize-10))

    captcha_text = random_captcha_text(captcha_size=textlen)
    captcha_text = ''.join(captcha_text)
    # captcha = image.generate(captcha_text)
    captcha = my_generate_image(image, captcha_text)
    captcha_image = Image.open(captcha)

    # 保存为文件
    captcha_image.save("training_data/captcha_img/" + captcha_text + ".png")

    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image


if __name__ == '__main__':
    ##展示验证码

    # fontslist = [r'C:\Windows\Fonts\Georgia.ttf', r'C:\Windows\Fonts\times.ttf', r'C:\Windows\Fonts\Calibri.ttf', r'C:\Windows\Fonts\Arial.ttf', r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simsunb.ttf']
    fontslist = [r'C:\Windows\Fonts\Georgia.ttf', r'C:\Windows\Fonts\times.ttf', r'C:\Windows\Fonts\Calibri.ttf', r'C:\Windows\Fonts\Arial.ttf', r'C:\Windows\Fonts\msyh.ttf', r'C:\Windows\Fonts\simsunb.ttf']

    for i in range(10000):
        textlen = random.randint(2, 10)
        fontsize = random.randint(80, 200)
        fontname = fontslist[random.randint(0, len(fontslist)-1)]
        print(i, textlen, fontsize, fontname)
        text, image = gen_captcha_text_and_image(textlen, fontsize, fontname)

        # f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        # plt.imshow(image)
        # plt.show()
