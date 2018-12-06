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
# 改变图片的resize方式，避免图片变形，采用填0或1方式！


# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import shutil
import pandas as pd
import cv2

IMAGE_LABEL = "image_label.txt"
LABEL = "label.txt"
RECORD_FILE = "captcha.tfrecord"
CHAR_DICT = "char_dict.txt"
MAX_LABEL_LEN = 30

# 不同的来源不同，生成image_label.txt：根据实际情况修改
def gen_image_label(img_path, output_path):
    full_img_label = os.path.join(output_path, IMAGE_LABEL)
    if os.path.isfile(full_img_label):
        print("%s exists, now deleted!" % full_img_label)
        os.remove(full_img_label)

    fp = open(full_img_label, 'w')
    file_list = os.listdir(img_path)      # 列出文件夹下所有的目录与文件

    for i in range(0, len(file_list)):
        filename = file_list[i]
        item = filename + " " + filename[0:4] + "\n"       # 头四位是标签(前闭后开)
        fp.write(item)

    print(full_img_label + " is generated!")
    fp.close()

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
    dict_list[x+1] = char
    return dict_list, str(x+1)


#
# 将图片文件转换TFRecord格式：
#
def gen_tfrecord(img_path, output_path):
    print("Converting data into tfrecord ...\n")

    # TFRecord文件是否存在
    full_rec_name = os.path.join(output_path, RECORD_FILE)
    if os.path.isfile(full_rec_name):
        print("%s exists, now deleted!" % full_rec_name)
        os.remove(full_rec_name)
    writer = tf.python_io.TFRecordWriter(full_rec_name)

    # char_dict.txt文件是否存在
    full_char_dict = os.path.join(output_path, CHAR_DICT)
    if os.path.isfile(full_char_dict):
        print("%s exists, now deleted!" % full_char_dict)
        os.remove(full_char_dict)

    char_dict = {}      # 采用字典方式

    full_label_file = os.path.join(output_path, LABEL)
    full_img_label = os.path.join(output_path, IMAGE_LABEL)

    fp = open(full_label_file, "w")
    img_label = open(full_img_label)

    for item in img_label:
        # print("ITEM=", item)
        item = item.rstrip("\n")
        spt = item.split(' ')           # 前面是文件名，后面是标签
        # images.append(os.path.join(img_path, spt[0]))
        img_name = os.path.join(img_path, spt[0])
        fp.write(spt[0])
        item_label = []

        # 将标签逐一转化为编码，同时写label.txt
        for x, ch in enumerate(spt[1]):  # item.content
            # print("CH=", ch)
            char_dict, code = get_char_code(char_dict, ch)      # 得到对应的编号ch[1]
            item_label.append(code)
            fp.write(" " + code)

        fp.write("\n")
        # labels.append(item_label)

        # 写TFRecord：
        img = Image.open(img_name)
        img = img.convert('L')
        print("img size: ", img.shape)
        # img = img.resize((280, 32))     # 调整大小为统一尺寸:
        img = img.resize((96, 32))     # 调整大小为统一尺寸:?
        print("img size convert to: ", img.shape)
        img_raw = img.tobytes()

        label = np.asarray(item_label, dtype=np.int64)
        print("LABEL=", label)

        if len(label) < MAX_LABEL_LEN:
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                # [img_raw]表示为[280,32,1],只考虑1个通道
            }))

        writer.write(example.SerializeToString())  # Serialize To String

    fp.close()
    writer.close()

    # 生成可能的标签字母表：
    dict_file = os.path.join(output_path, CHAR_DICT)

    fp = open(dict_file, 'w', encoding='utf-8')
    for x, word in enumerate(char_dict):
        if char_dict[x] != '\n':
            fp.write(char_dict[x])
            fp.write("\n")

    print("Convert finished：", dict_file, full_rec_name, full_label_file)
    return


#
# 读取TFRecord：需要知道打包时的格式
#
def read_tfrecord(recfile_path, recfile_name=RECORD_FILE, img_size=[32, 280], show_num=2):
    # TFRecord文件是否存在
    full_rec_name = os.path.join(output_path, recfile_name)
    print("Reading tfrecord：", full_rec_name)

    if not os.path.isfile(full_rec_name):
        print("%s not exists!" % full_rec_name)

    filename_queue = tf.train.string_input_producer([full_rec_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)     # return the file and the name of file

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
    img = tf.reshape(img, img_size)              # 如何填充0？显示图片是不转化为[32, 96, 1]

    print("IMG SIZE: ", img.shape)
    # print("IMG: ", img)
    init = tf.global_variables_initializer()

    # 显示文件：
    session = tf.Session()
    with session.as_default():
        session.run(init)

        # 创建线程并使用QueueRunner对象来提取数据, 使用tf.train函数添加QueueRunner到tensorflow中。
        # 在运行任何训练步骤之前，需要调用tf.train.start_queue_runners函数，否则tensorflow将一直挂起。
        coord = tf.train.Coordinator()                          # #创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)     # 启动QueueRunner, 此时文件名队列已经进队。

        for i in range(show_num):
            example, l = session.run([img, label])              # 在会话中取出image和label
            # print(example.shape)
            imgo = Image.fromarray(example)                     # 这里Image是之前提到的
            # imgo = Image.fromarray(example, cv2.CV_LOAD_IMAGE_GRAYSCALE)               # 这里Image是之前提到的
            # img.save('./' + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
            # imgo = imgo.covert('L')
            # 可以调用Image库下的函数了，比如show()
            imgo.show()

        coord.request_stop()
        coord.join(threads)

    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    return img, label


# 生成字典文件: 或者用ord(), chr()函数进行整数和中文转换，缺点是数字太大，不连续!
def CreateDict(dictList, dictFileName):
    fp = open(dictFileName, 'w', encoding='utf-8')
    for x, word in enumerate(dictList):
        if dictList[x] != '\n':
            fp.write(dictList[x])
            fp.write("\n")
    return

# 作废！

# 找到单个文字对应的编码：
def GetCharCode(dictList, char):
    x = 0

    if len(char) == 0:
        print("Error: char is none ", char)
        return None

    # 当dictList为空时：避免enumerate报错！
    if len(dictList) == 0:
        dictList.append(char)
        return 0

    for x, ch in enumerate(dictList):
        if dictList[x] == char:
            return x

    # 没有找到，则添加到末尾：
    dictList.append(char)
    # print("Not Found: [", char, "]")
    return x+1


# 作废！
#
# 添加图片文件名和内容到标签文件
# destdir\label.txt: item.filename item.content
#
def InsertLabelItem(dictList, item, destDir, lableFile):
    # print(filename, content)
    oneitem = item[0] + ".jpg"       # item.filename
    # full_lablefile = os.path.join(destDir, lableFile)
    fp = open(lableFile, "a+")
    
    for ch in enumerate(item[1]):       # item.content
        code = GetCharCode(dictList, ch[1])       # 得到对应的编号ch[1]
        oneitem = oneitem + " " + str(code)
        
    oneitem = oneitem + "\n" 
    fp.write(oneitem)
    fp.close()


# 作废！
#
# 将图片文件转换TFRecord格式：
# 作废！
#
def Data2TFRecord(picPath, lableName, recName):
    # TFRecord文件是否存在
    if os.path.isfile(recName):
        print("%s exists, now deleted!" % recName)
        os.remove(recName)

    labels = []
    images = []

    #for i,file_path in enumerate(file_path_list):
    train_txt = open(lableName)

    for idx in train_txt:
        idx = idx.rstrip("\n")
        spt = idx.split(' ')
        images.append(os.path.join(picPath, spt[0]))
        labels.append(spt[1:])
    
    #print("FILES: ", images[0:10])
    #print("LABEL[0]: ", labels[0:10])
    print("Converting data into %s ..." % recName)
    cwd = os.getcwd()

    writer = tf.python_io.TFRecordWriter(recName)

    for index, img_name in enumerate(images):
        img = Image.open(img_name)
        img = img.convert('L')

        # 调整大小为统一尺寸：
        img = img.resize((280, 32))
        img_raw = img.tobytes()

        label = np.asarray(labels[index], dtype=np.int64)
        print(label)
        
        if len(label) < 30: 
            example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),      # [img_raw]表示为[280,32,1],只考虑1个通道
                }))
        
            writer.write(example.SerializeToString())  # Serialize To String
        
    writer.close()
    
    print("Convert finished!")
    return



# 作废！
# 读取目录及子目录下所有文件：拷贝到指定文件夹
# 作废！
def GenTrainingData(dictList, picDir, destDir, lableFile, recName):
    # 删除目标目录下所有文件：
    if os.path.isfile(recName):
        print("%s exists, now deleted!" % recName)
        os.remove(recName)

    if os.path.isfile(lableFile):
        print("%s exists, now deleted!" % lableFile)
        os.remove(lableFile)
        

    #dataset = pd.read_csv("./data/problem3/content_train.csv", nrows=50)
    dataset = pd.read_csv("./data/problem3/content_train.csv")
    
    # 总行数：
    num = int(dataset.size/2)
    print("Total rows: ", num)
    
    filenames = dataset.iloc[ : , :].values
    
    for i in range(num):
        #print(filenames[i][0], filenames[i][1])
        InsertLabelItem(dictList, [filenames[i][0], filenames[i][1]], destDir, lableFile)
    
    # 生成代码表：
    CreateDict(dictList, './training_data/ocr_char.txt')
    
    # 生成TFRecord：
    Data2TFRecord('./data/problem3/train','./training_data/label.txt', './training_data/ocr.tfrecords')

    return


#
# Main:
#

dictList = []
# GenTrainingData(dictList, './data/problem3/train/', './training_data','./training_data/label.txt', './training_data/data.tfrecords')

img_path = './make_tfrecords/images/captcha_png'
output_path = './training_data'
# gen_image_label(img_path, output_path)

# 生成TFRecord文件：
# gen_tfrecord(img_path, output_path)

# 读取TFRecord并显示图片：show_num为显示的数量
read_tfrecord(output_path, 'ocr.tfrecords', show_num=40)
# read_tfrecord(output_path, img_size=[32, 96])



