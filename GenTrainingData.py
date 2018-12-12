#
# 生成训练数据(GenTrainingData)：大写的函数均作废！
#
# 输入： 文件夹images下为所有的图片文件（.jpg, .png), 以及配置文件image_label.txt（每个文件对应的标签）
# 输出： char_dict.txt(分类)，label.txt（每个文件标签对应的编码：文件名 编码1 编码2 ...）, tfrecord文件
#
# 其他功能：
# 根据tfrecord文件恢复图片文件，读取文件内容为训练输入等。
#
# 需要完善的地方：
# 1.改变图片的resize方式，避免图片变形，采用填0或1方式！
# 2.BUG: 当CSV文件标注可能是空，或者以空格开头！转换有问题！ (OK)
#


# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import cv2

# GPU配置：
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAINFILE_LABEL = "image_train_label.txt"
TESTFILE_LABEL = "image_test_label.txt"
TRAIN_LABEL = "train_label.txt"
TEST_LABEL = "test_label.txt"
TRAIN_RECORD_FILE = "captcha_ocr_train.tfrecords"
TEST_RECORD_FILE = "captcha_ocr_test.tfrecords"
CHAR_DICT = "char_dict.txt"


# 不同的来源不同，生成image_label.txt：
def gen_image_label(img_path, output_path, imagelabel):
    full_img_label = os.path.join(output_path, imagelabel)
    if os.path.isfile(full_img_label):
        print("%s exists, now deleted!" % full_img_label)
        os.remove(full_img_label)

    fp = open(full_img_label, 'w')
    file_list = os.listdir(img_path)  # 列出文件夹下所有的目录与文件

    for i in range(0, len(file_list)):
        filename = file_list[i]
        # item = filename + " " + filename[0:4] + "\n"  # 头四位是标签(前闭后开)
        item = filename + " " + filename[0:-4] + "\n"  # 文件名是标签
        fp.write(item)

    print(full_img_label + " is generated!")
    fp.close()

    return


# 针对OCR生成image_label.txt：根据CSV文件：
def gen_ocr_image_label(csv_file, output_path):
    full_img_label = os.path.join(output_path, TRAINFILE_LABEL)
    if os.path.isfile(full_img_label):
        print("%s exists, now deleted!" % full_img_label)
        os.remove(full_img_label)

    fp = open(full_img_label, 'w')
    dataset = pd.read_csv(csv_file)
    filenames = dataset.iloc[:, :].values

    for i in range(int(filenames.size / 2)):

        label = filenames[i][1].strip()

        if len(label) != 0:
            # print(i, filenames[i][0], filenames[i][1])
            item = filenames[i][0] + '.jpg ' + label + "\n"
        fp.write(item)

    fp.close()
    print(full_img_label + " is generated!")

    return


# 找到标签编码：返回新的dict_list, 位置
def get_char_code(dict_list, char):
    x = 0

    if len(char) == 0:
        print("Error: char is none ", char)
        return dict_list, "NONE"

    # 当dict_list为空时：避免enumerate报错！
    if len(dict_list) == 0:
        dict_list[0] = char
        return dict_list, "0"

    for x, ch in enumerate(dict_list):
        if dict_list[x] == char:
            return dict_list, str(x)

    # 没有找到，则添加到末尾：
    dict_list[x + 1] = char
    return dict_list, str(x + 1)


# 对图形文件进行转换，填充为指定大小：填充0或255
def image_convert(imgfile, conv_size=(32, 280), conv_color=[0, 0, 0]):

    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img.shape
    # print(height, width, conv_size)

    # 当输入图像很扁时，宽度调整为变换后的宽度，在下面填充：
    if height / width < conv_size[0] / conv_size[1]:
        ratio = conv_size[1] / width
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)  # 注意：直接指定是，与shape相反！是宽x高
        img = cv2.copyMakeBorder(img, 0, conv_size[0] - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=conv_color)
    else:
        # 当输入图像较为方正时，高度调整为变换后的高度，在右边填充：
        # 如果宽<<高，考虑需对图片进行旋转：
        ratio = conv_size[0] / height
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        img = cv2.copyMakeBorder(img, 0, 0, 0, conv_size[1] - img.shape[1], cv2.BORDER_CONSTANT, value=conv_color)

    # cv2.namedWindow("Image")
    # cv2.imshow("ImageC", img)
    # cv2.waitKey(0)

    return img


#
# 将图片文件转换TFRecord格式：
#
def gen_tfrecord(img_path, output_path, file_label=TRAINFILE_LABEL, rec_file=TRAIN_RECORD_FILE, img_label=TRAIN_LABEL, img_size=[280, 32], max_lablelen=30):
    print("Converting data into tfrecord ...\n")

    # TFRecord文件是否存在
    full_rec_name = os.path.join(output_path, rec_file)
    if os.path.isfile(full_rec_name):
        print("%s exists, now deleted!" % full_rec_name)
        os.remove(full_rec_name)
    writer = tf.python_io.TFRecordWriter(full_rec_name)

    # char_dict.txt文件是否存在
    full_char_dict = os.path.join(output_path, CHAR_DICT)
    if rec_file == TRAIN_RECORD_FILE:       # 要先执行TRAIN_RECORD_FILE, 这时才重新生成char_dict
        if os.path.isfile(full_char_dict):
            print("%s exists, now deleted!" % full_char_dict)
            os.remove(full_char_dict)

    char_dict = {}  # 采用字典方式

    full_label_file = os.path.join(output_path, img_label)
    full_img_label = os.path.join(output_path, file_label)

    fp = open(full_label_file, "w")
    img_label = open(full_img_label)

    for item in img_label:
        # print("ITEM=", item)
        item = item.rstrip("\n")
        spt = item.split(' ')  # 前面是文件名，后面是标签
        # images.append(os.path.join(img_path, spt[0]))
        img_name = os.path.join(img_path, spt[0])
        fp.write(spt[0])
        item_label = []

        # 将标签逐一转化为编码，同时写label.txt
        for x, ch in enumerate(spt[1]):  # item.content
            # print("CH=", ch)
            char_dict, code = get_char_code(char_dict, ch)  # 得到对应的编号ch[1]
            item_label.append(code)
            fp.write(" " + code)

        fp.write("\n")
        # labels.append(item_label)

        # 写TFRecord：
        # img = Image.open(img_name)
        img = image_convert(img_name, conv_size=[img_size[1], img_size[0]], conv_color=[255, 255, 255])

        # img = img.convert('L')
        # print("img size: ", img.shape)
        # img = img.resize(img_size)  # 调整大小为统一尺寸:?
        img_raw = img.tobytes()

        label = np.asarray(item_label, dtype=np.int64)
        # print("LABEL=", label)

        if len(label) < max_lablelen:
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                # [img_raw]表示为[280,32,1],只考虑1个通道
            }))
            writer.write(example.SerializeToString())  # Serialize To String

            img = Image.open(img_name)
            resized = img.resize((img.size[0]*2, img.size[1]))
            # img.show()
            resized.save("./temp.png")

            img = image_convert("./temp.png", conv_size=[img_size[1], img_size[0]], conv_color=[255, 255, 255])
            img_raw = img.tobytes()
            # label = np.asarray(item_label, dtype=np.int64)

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))

            writer.write(example.SerializeToString())  # Serialize To String

    fp.close()
    writer.close()

    # 生成可能的标签字母表：
    if rec_file == TRAIN_RECORD_FILE:       # 要先执行TRAIN_RECORD_FILE, 这是才重新生成char_dict
        dict_file = os.path.join(output_path, CHAR_DICT)

        fp = open(dict_file, 'w', encoding='utf-8')
        for x, word in enumerate(char_dict):
            if char_dict[x] != '\n':
                fp.write(char_dict[x])
                fp.write("\n")

        fp.close()
        print("Convert finished：", dict_file, full_rec_name, full_label_file)
    else:
        print("Convert finished：", full_rec_name, full_label_file)

    return


#
# 读取TFRecord：需要知道打包时的格式
#
def read_tfrecord(recfile_path, recfile_name=TRAIN_RECORD_FILE, img_size=[32, 280], show_num=2):
    # TFRecord文件是否存在
    full_rec_name = os.path.join(recfile_path, recfile_name)
    print("Reading tfrecord：", full_rec_name)

    if not os.path.isfile(full_rec_name):
        print("%s not exists!" % full_rec_name)

    filename_queue = tf.train.string_input_producer([full_rec_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return the file and the name of file

    # 使用其他方式？TF只是定义图？
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.VarLenFeature(tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    label = tf.cast(features['label'], tf.int64)

    img = tf.decode_raw(features['img_raw'], tf.uint8)  # notice the type of data
    # img = tf.reshape(img, [32, 96, 1])              # 如何填充0？
    img = tf.reshape(img, img_size)  # 如何填充0？显示图片是不转化为[32, 96, 1]

    # print("IMG SIZE: ", img.shape)
    # print("IMG: ", img)
    init = tf.global_variables_initializer()

    # 显示文件：
    session = tf.Session()
    with session.as_default():
        session.run(init)

        # 创建线程并使用QueueRunner对象来提取数据, 使用tf.train函数添加QueueRunner到tensorflow中。
        # 在运行任何训练步骤之前，需要调用tf.train.start_queue_runners函数，否则tensorflow将一直挂起。
        coord = tf.train.Coordinator()  # #创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。

        for i in range(show_num):
            example, l = session.run([img, label])  # 在会话中取出image和label
            # print(example.shape)
            imgo = Image.fromarray(example)  # 这里Image是之前提到的
            # imgo = Image.fromarray(example, cv2.CV_LOAD_IMAGE_GRAYSCALE)               # 这里Image是之前提到的
            # img.save('./' + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
            # imgo = imgo.covert('L')
            # 可以调用Image库下的函数了，比如show()
            imgo.show()

        coord.request_stop()
        coord.join(threads)

    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    return img, label


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'label': tf.VarLenFeature(tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    label = tf.cast(features['label'], tf.int64)
    img = tf.decode_raw(features['img_raw'], tf.uint8)

    return img, label

from sklearn.model_selection import train_test_split

# 读取TFReords数据，可进行训练，验证和测试数据划分！
def get_training_test_data(train_files, num_epochs=1, batch_size=1, shuffle=False, shuffle_buffer=20000, test_size=0.1):        # shuffle_buffer定义随机打乱数据时buffer的大小
    dataset = tf.data.TFRecordDataset(train_files)
    dataset = dataset.map(parser)

    if shuffle:
        print("shuffle")
        dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)                   # 是否打乱数据

    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        sess.run(iterator.initializer)
        image_list = []
        label_list = []
        while True:
            try:
                image, label = sess.run([image_batch, label_batch])
                # print(image, label)
                image_list.append(image)
                label_list.append(label)
            except tf.errors.OutOfRangeError:
                break

    X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test
#
# Main:
#

# dictList = []
def main():
    train_img_path = './training_data/train_img'
    test_img_path = './training_data/test_img'
    output_path = './training_data'

    # read_tfrecord(output_path, img_size=[32, 180], show_num=10)
    # read_tfrecord(output_path, recfile_name=TEST_RECORD_FILE, img_size=[32, 180], show_num=5)

    # read_tfrecord(output_path, img_size=[32, 180], show_num=5)
    record_name = output_path + "/" + TRAIN_RECORD_FILE
    #
    X_train, X_test, y_train, y_test = get_training_test_data(record_name, shuffle_buffer=50000)
    print(len(X_train), type(X_train[0]), X_train[0])
    img = X_train[8].reshape([32, 180])
    # img = 255-img
    imgo = Image.fromarray(img)
    imgo.show()

    print(len(y_train), type(y_train), y_train[8])
    print(len(X_test), type(X_test), X_test[0])
    print(len(y_test), type(y_test), y_test[0])
    exit(1)


    # 1.生成标签列表文件：
    # gen_image_label(train_img_path, output_path, TRAINFILE_LABEL)
    # gen_image_label(test_img_path, output_path, TESTFILE_LABEL)
    # csvfile = "./data/problem3/content_train.csv"
    # gen_ocr_image_label(csvfile, output_path)

    # 2. 生成TFRecord文件：
    # 训练集：
    gen_tfrecord(train_img_path, output_path, img_size=[180, 32])
    # 测试集：
    # gen_tfrecord(test_img_path, output_path, file_label=TESTFILE_LABEL, rec_file=TEST_RECORD_FILE, img_label=TEST_LABEL, img_size=[180, 32])

    # 3. 读取TFRecord并显示图片：show_num为显示的数量
    read_tfrecord(output_path, img_size=[32, 180], show_num=5)
    # read_tfrecord(output_path, recfile_name=TEST_RECORD_FILE, img_size=[32, 180], show_num=5)

    # image_convert('./training_data/captcha_img/0cODdY2jW7.png', conv_size=(32, 180), conv_color=[255, 255, 255])


if __name__ == '__main__':
    main()
