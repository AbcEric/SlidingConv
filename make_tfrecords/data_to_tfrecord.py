#
# 重新运行前，手工删除label.txt文件，是累加！
#

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import shutil

# 1根据字典文件生成字典：否则从char_dict生成文件！
def create_char_dict(char_dict, filename, way):
    if way == 1:
        char_dict = {}
        file = open(filename, 'r', encoding='utf-8').read().splitlines()
        index = 0
        for char in file:
            print(index, char)
            char_dict[index] = char[0]
            index += 1
        print("char_dict_first:", char_dict)
    else:
        file = open(filename, 'w', encoding='utf-8')
        #index = 0
        for x, word in enumerate(char_dict):
            if char_dict[x] != '\n':
                file.write(char_dict[x])
                file.write("\n")

    return char_dict

# 根据训练结果解析对应的字符串：result形如[12, 2323, 232, ...]
def adjust_label(char_dict, result):
    result_str = ""

    for x, char in enumerate(result):
        result_str += char_dict[char]

    return result_str

# 找到单个文字对应的编码：
# !!!!!! 原来的有误！
def get_charcode(char_dict, char):
    # label = open("label.txt").read().splitlines()
    # print(self.char_dict[0], self.char_dict[1], char)
    x = 0

    if len(char) == 0:
        print("Error: char is none ",char)
        return None

    # 当char_dict为空时：避免enumerate报错！
    if len(char_dict) == 0:
        char_dict[0] = char
        return 0

    for x, ch in enumerate(char_dict):
        # print(char_dict[x])
        if char_dict[x] == char:
            return x

    # !!!!APPEND没有找到，则添加：
    char_dict[x+1] = char
    print("Not Found: [", char, "]")
    return x+1

#
# 根据目录下图片文件名和内容添加到标签文件
# destdir\label.txt: item.filename item.content
#
def insert_label(char_dict, item, destdir, lablefile):
    # print(filename, content)
    oneitem = item[0]       # item.filename
    full_lablefile = os.path.join(destdir, lablefile)
    fd = open(full_lablefile, "a+")

    for ch in enumerate(item[1]):       # item.content
        if ch[1] != "\n":      # 去掉末尾的换行符
            code = get_charcode(char_dict, ch[1])       # 得到对应的编号ch[1]
            oneitem = oneitem + " " + str(code)

    oneitem = oneitem + "\n"
    fd.write(oneitem)
    fd.close()

""" Save data into TFRecord """
def data_to_tfrecord(images, labels, filename):
    if os.path.isfile(filename):
        print("%s exists" % filename)
        os.remove(filename)

    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()

    writer = tf.python_io.TFRecordWriter(filename)

    for index, img_name in enumerate(images):
        img = Image.open(img_name)
        img = img.convert('L')

        # 调整大小为统一尺寸：
        img = img.resize((280, 32))
        img_raw = img.tobytes()

        label = np.asarray(labels[index], dtype=np.int64)

        example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()

# ：
def read_data_from_paths(file_path, name_list):
    labels = []
    file_names = []

    #for i,file_path in enumerate(file_path_list):
    file_name = os.path.join(file_path, name_list)
    train_txt = open(file_name)

    for idx in train_txt:
        idx = idx.rstrip("\n")
        spt = idx.split(' ')
        file_names.append(os.path.join(file_path, spt[0]))
        labels.append(spt[1:])

    return file_names, labels

import hashlib

def get_md5(url):
    """
    由于hash不处理unicode编码的字符串（python3默认字符串是unicode）
        所以这里判断是否字符串，如果是则进行转码, 初始化md5、将url进行加密、然后返回加密字串
    """
    if isinstance(url, str):
        url = url.encode("utf-8")
    md = hashlib.md5()
    md.update(url)
    return md.hexdigest()

# 包括子目录：
def read_allfile_from_path(char_dict, rootdir, destdir, lablename):
    _files = []
    list = os.listdir(rootdir)      # 列出文件夹下所有的目录与文件

    for i in range(0, len(list)):
        filename = list[i]
        filewithpath = os.path.join(rootdir, list[i])

        if os.path.isdir(filewithpath):
            _files.extend(read_allfile_from_path(char_dict, filewithpath, destdir, labelname))

        if os.path.isfile(filewithpath):
            if filewithpath.find(".jpg") >= 0 or filewithpath.find(".png") >= 0:
                # 有重名文件：唯一编码文件名！
                print(filewithpath)
                copyfile = get_md5(filewithpath) + ".jpg"
                copyfilewithpath = os.path.join(destdir, copyfile)
                i += 1
                print("FILE: ", list[i], copyfile)
                content = open(os.path.join(rootdir, list[i])).read()
                print("CONTENT: ", content)

                if len(content) < 30:       # 占只处理10以内的长度
                    insert_label(char_dict, [copyfile, content], destdir, lablename)

                    # 拷贝文件：
                    shutil.copyfile(filewithpath, copyfilewithpath)
                    _files.append(filename)

    return _files


import random
if __name__=="__main__":
    #char_dict = {}     # 换成字典dict更方便，dictInfo={"1":"a", "2":")"...)? dictInfo["1"]?
    char_dict = {}
    #char_dict = create_char_dict(char_dict, os.path.join("../", "char_std_5990.txt"), 1)
    #char_dict = create_char_dict(char_dict, os.path.join("./images", "char_word.txt"), 1)
    #print("1: ", get_charcode(char_dict, '1'))
    #print("a: ", get_charcode(char_dict, 'a'))
    #print("1: ", get_charcode(char_dict, '1'))
    #str1 = adjust_label(char_dict, [29, 30, 14, 12, 21, 30, 7, 24, 3])
    #print(str1)
    labelname = "label.txt"
    file_path = "./images/"

    file_names = read_allfile_from_path(char_dict, "./images/input/", "./images/output/", labelname)

    print("READ:", file_names)
    create_char_dict(char_dict, os.path.join("./images", "char_word.txt"), 0)
    print("Char_dict_end:", char_dict)

    file_names, labels = read_data_from_paths("./images/output", labelname)

    print("FILE: ", file_names)
    print("LABEL:", labels[0])      # 太长！

    '''
    #-----shuffle------
    filename_label=[]
    for i in range(len(file_names)):
        filename_label.append(i)
    random.shuffle(filename_label)

    print("FILE-LABEL: ", filename_label)
    file_names_new=[]
    labels_new=[]

    for i in filename_label:
        file_names_new.append(file_names[i])
        labels_new.append(labels[i])

    # -----shuffle------
    print("shuffle OK！")
    print("FILE: ",file_names_new[0:5])
    print("LABLE: ",labels_new[0:5])
    data_to_tfrecord(file_names_new, labels_new, tfrecord_name)
    '''
    tfrecord_name = "data.tfrecords"
    print(file_names, labels)
    data_to_tfrecord(file_names, labels, tfrecord_name)
    #insert_label(char_dict, ["f0fe9157b8c0fe525d51e99eae983ac2.jpg", "(Step 1)"], "./images/output/", labelname)
