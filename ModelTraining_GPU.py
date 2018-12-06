#
# OCR识别(GPU)：采用Sliding Convolution
#
# 使用方法：
# 1.根据样本调整参数，包括：样本的类别总数classnum（实际种类+1），图片宽度，字符的长宽等。一开始要设置好，训练后不能更改！
# 2.相关数据：./training_data下char_dict.txt、label.txt和ocr.tfrecords。注意文本格式的差异：
#   Windows下文本文件在每一行末尾有一个回车和换行，用0x0D和0x0A两个字符("\r\n")表示；而UNIX文本只有一个换行，0x0A表示换行("\n")。
#   暂用notepad++进行格式转换，或者在生成时考虑格式；
#
# 修改历史：
# 1.lable.txt中标签的长度不能过长：tf.nn.ctc_loss会遇到问题，最大长度由(输入图片宽度-字符宽度)/移动距离决定;
#   太长也会导致ctc_loss出现"No valid path found"
# 2.GPU设置：
# 3.leariningrate设置：指数方式，先快后慢，设置初始值、系数等
# 4.转换程序：GenTrainingData

# 待完善：
# 1.增加验证数据集、epoch等：判断过拟合的时机。
# 2.如何多线程？
# 3.在GPU环境下需要shutdown进程才能重新运行？


# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import heapq
import time
from math import *
from PIL import Image
import csv

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TRAINING = False  # 训练还是预测


# tf.app.flags.DEFINE_string('mode', 'test', 'train or test')     # 重复执行有问题？ 好处是运行可带参数 --mode xxx


#
# 神经网络定义：
#
def sliding_generate_batch_layer(inputs, character_width=32, character_step=8):
    # inputs: batches*32*280*1

    for b in range(inputs.shape[0]):
        batch_input = inputs[b, :, :, :].reshape((1, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        for w in range(0, batch_input.shape[2] - character_width, character_step):
            if w == 0:
                output_batch = batch_input[:, :, w:(w + 1) * character_width, :]
            else:
                output_batch = np.concatenate((output_batch, batch_input[:, :, w:w + character_width, :]), axis=0)
        if b == 0:
            output = output_batch
        else:
            output = np.concatenate((output, output_batch), axis=0)

    return output


def network(batch_size=1, class_num=5990, character_height=32, character_width=32, character_step=8, is_train=False):
    network = {}

    # 输入参数："inputs"为img，注意seq_len的决定的lable标签最大长度
    network["inputs"] = tf.placeholder(tf.float32, [batch_size, 32, None, 1], name='inputs')
    network["seq_len"] = tf.multiply(tf.ones(shape=[batch_size], dtype=tf.int32),
                                     tf.floordiv(tf.shape(network["inputs"])[2] - character_width, character_step))

    network["inputs_batch"] = tf.py_func(sliding_generate_batch_layer,
                                         [network["inputs"], character_width, character_step], tf.float32)

    network["inputs_batch"] = tf.reshape(network["inputs_batch"], [-1, character_height, character_width, 1])

    network["conv1"] = tf.layers.conv2d(inputs=network["inputs_batch"], filters=50, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["batch_norm1"] = tf.contrib.layers.batch_norm(
        network["conv1"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm1"] = tf.nn.relu(network["batch_norm1"])

    network["conv2"] = tf.layers.conv2d(inputs=network["batch_norm1"], filters=100, kernel_size=(3, 3), padding="same",
                                        activation=tf.nn.relu)
    network["dropout2"] = tf.layers.dropout(inputs=network["conv2"], rate=0.1)

    network["conv3"] = tf.layers.conv2d(inputs=network["dropout2"], filters=100, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout3"] = tf.layers.dropout(inputs=network["conv3"], rate=0.1)
    network["batch_norm3"] = tf.contrib.layers.batch_norm(
        network["dropout3"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm3"] = tf.nn.relu(network["batch_norm3"])
    network["pool3"] = tf.layers.max_pooling2d(inputs=network["batch_norm3"], pool_size=[2, 2], strides=2)

    network["conv4"] = tf.layers.conv2d(inputs=network["pool3"], filters=150, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout4"] = tf.layers.dropout(inputs=network["conv4"], rate=0.2)
    network["batch_norm4"] = tf.contrib.layers.batch_norm(
        network["dropout4"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm4"] = tf.nn.relu(network["batch_norm4"])

    network["conv5"] = tf.layers.conv2d(inputs=network["batch_norm4"], filters=200, kernel_size=(3, 3), padding="same",
                                        activation=tf.nn.relu)
    network["dropout5"] = tf.layers.dropout(inputs=network["conv5"], rate=0.2)

    network["conv6"] = tf.layers.conv2d(inputs=network["dropout5"], filters=200, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout6"] = tf.layers.dropout(inputs=network["conv6"], rate=0.2)
    network["batch_norm6"] = tf.contrib.layers.batch_norm(
        network["dropout6"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm6"] = tf.nn.relu(network["batch_norm6"])
    network["pool6"] = tf.layers.max_pooling2d(inputs=network["batch_norm6"], pool_size=[2, 2], strides=2)

    network["conv7"] = tf.layers.conv2d(inputs=network["pool6"], filters=250, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout7"] = tf.layers.dropout(inputs=network["conv7"], rate=0.3)
    network["batch_norm7"] = tf.contrib.layers.batch_norm(
        network["dropout7"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm7"] = tf.nn.relu(network["batch_norm7"])

    network["conv8"] = tf.layers.conv2d(inputs=network["batch_norm7"], filters=300, kernel_size=(3, 3), padding="same",
                                        activation=tf.nn.relu)
    network["dropout8"] = tf.layers.dropout(inputs=network["conv8"], rate=0.3)

    network["conv9"] = tf.layers.conv2d(inputs=network["dropout8"], filters=300, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout9"] = tf.layers.dropout(inputs=network["conv9"], rate=0.3)
    network["batch_norm9"] = tf.contrib.layers.batch_norm(
        network["dropout9"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm9"] = tf.nn.relu(network["batch_norm9"])
    network["pool9"] = tf.layers.max_pooling2d(inputs=network["batch_norm9"], pool_size=[2, 2], strides=2)

    network["conv10"] = tf.layers.conv2d(inputs=network["pool9"], filters=350, kernel_size=(3, 3), padding="same",
                                         activation=None)
    network["dropout10"] = tf.layers.dropout(inputs=network["conv10"], rate=0.4)
    network["batch_norm10"] = tf.contrib.layers.batch_norm(
        network["dropout10"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm10"] = tf.nn.relu(network["batch_norm10"])

    network["conv11"] = tf.layers.conv2d(inputs=network["batch_norm10"], filters=400, kernel_size=(3, 3),
                                         padding="same",
                                         activation=tf.nn.relu)
    network["dropout11"] = tf.layers.dropout(inputs=network["conv11"], rate=0.4)

    network["conv12"] = tf.layers.conv2d(inputs=network["dropout11"], filters=400, kernel_size=(3, 3), padding="same",
                                         activation=None)
    network["dropout12"] = tf.layers.dropout(inputs=network["conv12"], rate=0.4)
    network["batch_norm12"] = tf.contrib.layers.batch_norm(
        network["dropout12"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm12"] = tf.nn.relu(network["batch_norm12"])
    # 2*2*400
    network["pool12"] = tf.layers.max_pooling2d(inputs=network["batch_norm12"], pool_size=[2, 2], strides=2)

    network["flatten"] = tf.contrib.layers.flatten(network["pool12"])

    network["fc1"] = tf.contrib.layers.fully_connected(inputs=network["flatten"], num_outputs=900,
                                                       activation_fn=tf.nn.relu)
    network["dropout_fc1"] = tf.layers.dropout(inputs=network["fc1"], rate=0.5)
    network["fc2"] = tf.contrib.layers.fully_connected(inputs=network["dropout_fc1"], num_outputs=200,
                                                       activation_fn=tf.nn.relu)
    if is_train:
        network["fc3"] = tf.contrib.layers.fully_connected(inputs=network["fc2"], num_outputs=class_num,
                                                           activation_fn=None)
    else:
        network["fc3"] = tf.contrib.layers.fully_connected(inputs=network["fc2"], num_outputs=class_num,
                                                           activation_fn=tf.nn.sigmoid)
    network["outputs"] = tf.reshape(network["fc3"], [batch_size, -1, class_num])
    network["outputs"] = tf.transpose(network["outputs"], (1, 0, 2))

    return network


#
# 读入TFRecord数据并解码：
#
def read_and_decode(fileName):
    # generate a queue with a given file name
    print("reading tfrecords from {}".format(fileName))

    filename_queue = tf.train.string_input_producer([fileName])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return the file and the name of file

    features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                       features={
                                           'label': tf.VarLenFeature(tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    label = tf.cast(features['label'], tf.int64)
    img = tf.decode_raw(features['img_raw'], tf.uint8)  # notice the type of data

    # 比较一下前后是否一致
    img = tf.reshape(img, [32, 280, 1])  # 为何变为[32, 280]?  因为全连接层需要固定大小的输入，才能最终要得到固定维度的输出，
    # 而卷积层不会改变图片大小，池化层会固定降低图片维度，所以需要reshape输入图片
    # 如果网络结构中不包含全连接层，则不要求图片的输入维度，如全卷积神经网络等。

    # 归一化？
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img, label


# 神经网络设置及方法定义：
class SlidingConvolution(object):
    # 初始设置：
    def __init__(self, is_train=False):
        # self.class_num = 5990           # 要识别的分类类别
        self.class_num = 3470  # 分类类别:数字、字母和常见符号，应与ocr_char.txt中的数量+1(? 否则可能报错)！
        self.character_step = 8  # ？移动的距离，和输入图片宽度、字符宽度共同确定可以识别的最大字符数量，
        # 如；（280-32）/8 = 31。过小训练量加大！
        self.character_height = 32  # 转换为280*32固定大小，若文字高度本身很小呢？
        self.character_width = 32  #
        # self.log=open('./saving_model/2018-11-19/train_log/trainlog-%s' % time.strftime("%Y-%m-%d-%I-%M-%S"), 'w+')

        # 训练设置：
        self.train_tfrecords_name = "./training_data/ocr.tfrecords"  # 训练数据路径
        self.train_ocrchar_name = "./training_data/char_dict.txt"  # 训练数据路径!!!
        self.summary_save_path = "./saving_model/"  # 可以一个目录
        self.summary_steps = 2000
        self.save_steps = 500
        self.save_path = "./saving_model"
        self.best_save_path = "./saving_model/best_models/"

        # 测试设置:
        self.model_path = "./saving_model/sliding_conv.ckpt-20"

        if is_train:
            self.batch_size = 64
            self.with_clip = True  # 采用Gradient Clipping
            self.network = network(batch_size=self.batch_size, class_num=self.class_num,
                                   character_height=self.character_height, character_width=self.character_width,
                                   character_step=self.character_step, is_train=True)
        else:
            # 预测：
            current_path = os.path.dirname(os.path.abspath('__file__'))

            # 字典对照表
            self.char_dict = self.create_char_dict(
                # os.path.join(current_path, "char_word.txt))
                self.train_ocrchar_name)  # 样本中的所有字符

            self.batch_size = 1
            self.graph = tf.Graph()

            # 需了解环境的情况：
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                                      graph=self.graph)

            with self.session.as_default():
                with self.graph.as_default():
                    self.network = network(batch_size=self.batch_size, class_num=self.class_num,
                                           character_height=self.character_height, character_width=self.character_width,
                                           character_step=self.character_step, is_train=False)

                    # ct_greedy_decoder根据当前序列预测下一个字符，并且取概率最高的作为结果，再此基础上再进行下一次预测。
                    # 输入：
                    #     inputs: 一个3-D Tensor (max_time * batch_size * num_classes),保存着logits(通常是RNN接上一个输出)
                    #     sequence_length: 1-D int32 向量, size为 [batch_size].序列的长度.
                    #     merge_repeated: Default: True.
                    # 输出：
                    #     decoded: decoded[0]是SparseTensor,保存着解码的结果, 分别为索引矩阵、值向量和shape！.
                    self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.network["outputs"],
                                                                           self.network["seq_len"],
                                                                           merge_repeated=True)

                    init = tf.global_variables_initializer()

                    self.session.run(init)
                    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

                    # 恢复保存的模型：
                    print("restore model: ", self.model_path)
                    saver.restore(self.session, self.model_path)

    # 根据字典文件生成字典：可以用char_dict.values()得到Value, char_dict.key()得到Key。{Key1：Value1，Key2：Value2，...}
    # 可以显性指定Key，否则缺省用数字。list(char_dict.keys())[list(char_dict.values()).index("ValueFind")]可找到Value对应的key
    def create_char_dict(self, filename):
        char_dict = {}
        file = open(filename, 'r', encoding='utf-8').read().splitlines()
        index = 0
        for char in file:
            char_dict[index] = char[0]
            index += 1
        return char_dict

    # 根据训练结果解析对应的字符串：
    def adjust_label(self, result):
        result_str = ""
        #         print(self.char_dict)
        # print("Result(before):", result)
        for x, char in enumerate(result):
            # result_str += self.char_dict[char]
            result_str += self.char_dict[char]

        return result_str

    # 模型训练：
    def train_model(self):
        # 读取并解码数据：
        train_data, train_label = read_and_decode(self.train_tfrecords_name)

        # train_inputs是Tensor,每个元素为（32,280,1），targets是稀疏张量（SparseTensor），高效存储很多为0的数据。
        # 每次生成指定batch_size个的数据：
        train_inputs, train_targets = tf.train.shuffle_batch([train_data, train_label],
                                                             batch_size=self.batch_size, capacity=2000,
                                                             min_after_dequeue=1000)

        global_step = tf.Variable(0, trainable=False)

        # 学习率设置：
        learning_rate = tf.train.exponential_decay(0.003,
                                                   global_step,
                                                   2000,
                                                   0.9,
                                                   staircase=True)
        # SparseTensor:
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # 损失函数：处理输出标签预测值network["outputs"]和真实标签之间的损失
        # 经过RNN后输出的标签预测值为3D浮点Tensor，默认形状为max_time * batch_size * num_classes，max_time？
        # sequence_length: 1-D int32 vector, 大小为[batch_size]，vector中的每个值表示序列的长度.
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=targets, inputs=self.network["outputs"], sequence_length=self.network["seq_len"]))
        tf.summary.scalar("loss", loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if update_ops:
            with tf.control_dependencies([tf.group(*update_ops)]):
                if self.with_clip:
                    # Gradient Clipping的引入是为了处理梯度爆炸或梯度消失的问题，让权重更新限制在一个合适的范围。
                    print("Gradient Clipping")
                    tvars = tf.trainable_variables()

                    # 修正梯度值：由于链式求导，导致梯度指数级衰减！第一个参数是待修剪的张量，第二是修剪比例。
                    # 返回：grads是修剪后的张量，norm是一个中间计算量
                    grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
                    optimizer_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
                else:
                    optimizer_op = optimizer.minimize(loss, global_step=global_step)
        else:
            if self.with_clip:
                tvars = tf.trainable_variables()
                grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
                optimizer_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
            else:
                optimizer_op = optimizer.minimize(loss, global_step=global_step)

        # 预测时也要采用如下定义:
        decoded, log_prob = tf.nn.ctc_greedy_decoder(self.network["outputs"], self.network["seq_len"],
                                                     merge_repeated=True)
        acc = 1 - tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        tf.summary.scalar("accuracy", acc)
        merge_summary = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)  # 作用？
        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
            session.run(init)
            # 作用？？
            threads = tf.train.start_queue_runners(sess=session)

            # 之前max_to_keep设置为保留20轮，所以导致模型丢失,此参数默认值为5
            #             saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

            saver.restore(session, self.model_path)  # 恢复保存的模型，继续训练！
            summary_writer = tf.summary.FileWriter(self.summary_save_path, session.graph)

            while True:
                step_input, step_target, steps, lr = session.run(
                    [train_inputs, train_targets, global_step, learning_rate])

                feed = {self.network["inputs"]: step_input, targets: step_target}

                batch_acc, batch_loss, _ = session.run([acc, loss, optimizer_op], feed)

                log_out = "Step is: {}, batch loss is: {} learningrate is {}， acc is {}".format(steps, batch_loss, lr,
                                                                                                batch_acc)
                print(log_out)
                # print(log_out, file=self.log)
                max = 0.4
                if steps > 0 and steps % self.summary_steps == 0:
                    _, batch_summarys = session.run([optimizer_op, merge_summary], feed)
                    summary_writer.add_summary(batch_summarys, steps)

                if steps > 0 and steps % self.save_steps == 0:
                    save_path = saver.save(session, os.path.join(self.save_path, "sliding_conv.ckpt"),
                                           global_step=steps)
                    print(save_path)

    # 对图形文件进行转换，填充为指定大小：填充0或255
    def image_convert(self, imgfile, conv_size=(32, 280), conv_color=[0, 0, 0]):
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
        cv2.imshow("ImageC", img)
        cv2.waitKey(0)

        return img

    # 测试：
    def predict_model(self, input_data):
        #         if input_data.ndim == 3:
        #             input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)

        #         height, width = input_data.shape
        #         print("Photo Size: ", height, width)
        #         ratio = height / 32.0
        #         input_data = cv2.resize(input_data,(int(width / ratio),32))
        #         input_data = input_data.reshape((1, 32, int(width / ratio), 1))
        #         scaled_data = np.asarray(input_data / np.float32(255) - np.float32(0.5))
        img = cv2.imread(input_data, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow(input_data, img)
        # cv2.waitKey(1)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape

        if width < height:
            degree = 90
            heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
            widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
            matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
            matRotation[0, 2] += (widthNew - width) / 2
            matRotation[1, 2] += (heightNew - height) / 2
            img = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
            width = widthNew
            height = heightNew
        ratio = height / 32.0
        #########################################
        cv2.imwrite('./tmp.jpg', img, params=None)
        #         print(width,height)
        # plt.imshow(img,cmap='gray')

        width_new = width
        if width > 32:
            tmp = height / 32
            width_new = int(width / tmp)

        img2 = Image.open('./tmp.jpg').convert('RGBA')
        img2 = img2.resize((int(width_new), 32))
        img2 = img2.convert('L')
        a4im = Image.new('RGB',
                         (280, 32),  # A4 at 72dpi
                         (0, 0, 0))  # White
        a4im.paste(img2, img2.getbbox())
        a4im = a4im.convert('L')
        a4im.save('./tmp2.jpg', 'jpeg')
        ############################################
        #         img = cv2.resize(img,(280,32))
        img3 = cv2.imread('./tmp2.jpg', cv2.IMREAD_GRAYSCALE)
        height, width = img3.shape
        #         print(height,width)
        cv2.imshow(input_data, img3)
        cv2.waitKey(2)
        img3 = img3.reshape((1, 32, 280, 1))

        scaled_data = np.asarray(img3 / np.float32(255) - np.float32(0.5))
        #         cv2.imwrite('./tmp.jpg', img, params=None)
        #         # plt.imshow(img,cmap='gray')

        #         img2 = Image.open('./tmp.jpg').convert('RGB')#直接转L模式有错？
        #         img2 = img2.convert('L')
        #         img2 = img2.resize((280, 32))
        #         img_raw = img2.tobytes()
        #         scaled_data = np.asarray(img / np.float32(255) - np.float32(0.5))

        with self.session.as_default():
            with self.graph.as_default():
                feed = {self.network["inputs"]: scaled_data}

                # outputs为ndarray，其长度可变，随预测文字的长度而变化。decoded为SparseTensorValue:
                # 应该用outputs值！ decoded是预测的下一个值！
                outputs, decoded = self.session.run([self.network["outputs"], self.decoded[0]], feed)

                predict = []
                for k in range(outputs.shape[0]):  # 很多字母的概率都很大，很难区分出来！
                    # 分别找到概率最大和最小的位置，从而可找到相应的字母
                    pos = list(map(list(outputs[k][0]).index, heapq.nlargest(1, list(outputs[k][0]))))
                    predict.append(pos[0])
                #                     print(np.max(outputs[k][0]), np.min(outputs[k][0]))
                #                     if k == 0:
                #                         print(outputs[k][0])
                # for j in range(outputs[k][0].size):
                # if outputs[k][0][j] > 0.8:
                # print(j, outputs[k][0][j])

                #                 print("OUTPUTS:", type(outputs), outputs.size, outputs.shape)
                #                 print("PdddREDICT:", predict)
                #                 print("DECODED:", type(decoded), decoded[1])                    # decoded[0]为Sparse值的位置, [1]为值，[2]为shape

                result_str = self.adjust_label(decoded[1])
                return result_str


def main():
    # if tf.app.flags.FLAGS.mode == "train":
    if TRAINING:
        print("Training ....")
        # 使用gpu训练
        with tf.device('/device:GPU:0'):
            sc = SlidingConvolution(is_train=True)
            sc.train_model()
    else:
        print("Predicting ....")
        # image = cv2.imread("./data/problem3/train/1fa279bd46ee4781bcb5816686f0f6d8.jpg", 1)
        sc = SlidingConvolution(is_train=False)
        # sc.image_convert('./make_tfrecords/images/captcha_png/1.jpg')
        sc.image_convert('./make_tfrecords/images/captcha_png/1.jpg', conv_size=(32, 120), conv_color=[255, 255, 255])

        exit(1)

        rootdir = './make_tfrecords/images/captcha_png'
        list = os.listdir(rootdir)

        with tf.device('/device:GPU:0'):
            print(len(list))
            for i in range(0, 6):
                path = os.path.join(rootdir, list[i])
                if os.path.isfile(path):
                    result_str = sc.predict_model(path)
                    print(i)
                    print("path" + path)
                    print("result = [" + result_str + "]")


#         sc = SlidingConvolution(is_train=False)
#         result_str = sc.predict_model("./data/problem3/train/1ff7c3142f9b42588a4a8d5c31a9d3d4.jpg")
#         print("result = ["+result_str+"]")


if __name__ == '__main__':
    # tf.app.run()       # main() takes 0 positional arguments but 1 was given
    main()
