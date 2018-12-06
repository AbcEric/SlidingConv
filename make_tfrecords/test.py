
import os
import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from model import network
from data_gen import read_and_decode
import tensorflow as tf

from data_gen import read_and_decode

def gen_tfrecord():
    tfrecord_name = "data.tfrecords"
    print("Writing ....")
    #data_to_tfrecord(["./make_tfrecords/images/output/7afd99ab6357cd274093011564315e93.jpg", \
    #                  "./make_tfrecords/images/output/e13c79f8622c2ed0e18da98c2281f39a.jpg"], [['9','3','7','2','20', '1','1','1','5'], ['1', '2', '7']], tfrecord_name)

    print("Reading ....")
    img, label = read_and_decode(tfrecord_name)
    print(img, label)
    print("SIZE: ", type(label))
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=1, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.global_variables_initializer()

    with tf.Session() as ss:
        ss.run(init)
        tf.train.start_queue_runners(sess=ss)
        print("LABEL: ", ss.run(label_batch))
        #print("IMG: ", ss.run(img_batch))


def img_convert(img_name):
    img = Image.open(img_name)
    img = img.convert('L')

    print(img.size, img, img.mode)
    # img.show()

    # 调整大小为统一尺寸：(280, 32)
    xy_ratio = img.size[0]/img.size[1]
    x_scale = img.size[0]/280
    y_scale = img.size[1]/32

    print(xy_ratio, x_scale, y_scale)

    # Image的size的宽x高！而转换为array后变成高x宽
    if ( xy_ratio < 280/32):        # 比较方正
        img = img.resize((int(img.size[0]/y_scale), 32))
        print(img.size)
        # 右边补0：
        img_array = np.asarray(img)
        print(img_array.shape)

        zero_img = np.zeros((32, 280-img_array.shape[1]))       # 0:为黑色
        print(img_array.shape, zero_img.shape)
        img_array = np.append(img_array, zero_img, axis=1)
        img_con = 255 - img_array

        print(img_array.shape)
        img = Image.fromarray(img_array)
        img.show()
        img_conv = Image.fromarray(img_con)
        img_conv.show()

        # 反转颜色
        inv_img = PIL.ImageOps.invert(img)
        # inv_img.show()


    else:                           # 很长
        img = img.resize((280, int(img.size[1] / x_scale)))
        # 下边补0：

        # img = img.resize((280, 32))
        print(type(img), img)
        img.show()

    img_raw = img.tobytes()

    return

if __name__ == '__main__':
    # gen_tfrecord()
    #img_convert("./images/captcha_png/char6.png")
    # img_convert("./images/captcha_png/2.jpg")
    img_convert("./images/captcha_png/4TV0_1530629973.png")