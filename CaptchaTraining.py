#
# Captcha识别：采用Sliding Convolution
#
# Attention：
# 1.lable.txt中标签的长度不能过长：tf.nn.ctc_loss会遇到问题，最大长度由(输入图片宽度-字符宽度)/移动距离决定;
#   太长也会导致ctc_loss出现"No valid path found"
# 2.根据样本调整参数，包括：样本的类别总数（过大会导致部分字符的权重极高，很难判断是哪一个？），图片宽度，字符的长宽等。
#   一开始需要设置好，训练后不能更改！
# 3.相关数据：./training_data/ocr_char.txt、label.txt和ocr.tfrecords。注意文本格式的差异：
#   Windows下文本文件在每一行末尾有一个回车和换行，用0x0D和0x0A两个字符("\r\n")表示；而UNIX文本只有一个换行，0x0A表示换行("\n")。
#   暂用notepad++进行格式转换，或者在生成时考虑格式；


# Q:
# 1.CUDA的使用？GPU型号？TF中如何设置？
# 2.滑动窗口用于目标检测：对每个滑动区域预测目标出现的概率，计算成本高！窗口过大会影响性能，过小计算量大。
# 3.如何多线程？
# 4.leariningrate先快后慢：指数型（OK）
# 5.CTC？
# 6.转换程序
# 7.class_num为何数量要加1？

# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import heapq

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOAD_MODEL = True
# TRAINING = True            # 训练还是预测
TRAINING = False            # 训练还是预测
# tf.app.flags.DEFINE_string('mode', 'test', 'train or test')     # 重复执行有问题？ 好处是运行可带参数 --mode xxx


#
# 神经网络定义：
#
def sliding_generate_batch_layer(inputs,character_width=32, character_step=8):
    # inputs: batches*32*280*1

    for b in range(inputs.shape[0]):
        batch_input = inputs[b, :, :, :].reshape((1, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        for w in range(0, batch_input.shape[2]-character_width, character_step):
            if w == 0:
                output_batch = batch_input[:, :, w:(w+1)*character_width, :]
            else:
                output_batch = np.concatenate((output_batch, batch_input[:, :, w:w+character_width, :]), axis=0)

        if b == 0:
            output = output_batch
        else:
            output = np.concatenate((output,output_batch),axis=0)
        
    return output


def train_network(batch_size=1, class_num=5990, character_height=32, character_width=32, character_step=8, is_train=False):
    network = {}

    # 输入参数："inputs"为img，注意seq_len的决定的lable标签最大长度
    # CNN输入：4D向量，第2、3维对应图片的宽高，最后一维代表图片的颜色通道数，高为None？
    network["inputs"] = tf.placeholder(tf.float32, [batch_size, 32, None, 1], name='inputs')
    network["seq_len"] = tf.multiply(tf.ones(shape=[batch_size], dtype=tf.int32), tf.floordiv(tf.shape(network["inputs"])[2] - character_width, character_step))

    network["inputs_batch"] = tf.py_func(sliding_generate_batch_layer, [network["inputs"], character_width, character_step], tf.float32)

    network["inputs_batch"] = tf.reshape(network["inputs_batch"], [-1, character_height, character_width, 1])

    network["conv1"] = tf.layers.conv2d(inputs=network["inputs_batch"], filters=50, kernel_size=(3, 3), padding="same", activation=None)
    network["batch_norm1"] = tf.contrib.layers.batch_norm(
        network["conv1"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm1"] = tf.nn.relu(network["batch_norm1"])
    
    network["conv2"] = tf.layers.conv2d(inputs=network["batch_norm1"], filters=100, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    network["dropout2"] = tf.layers.dropout(inputs=network["conv2"], rate=0.1)
    
    network["conv3"] = tf.layers.conv2d(inputs=network["dropout2"], filters=100, kernel_size=(3, 3), padding="same", activation=None)
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
    
    network["conv11"] = tf.layers.conv2d(inputs=network["batch_norm10"], filters=400, kernel_size=(3, 3), padding="same",
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
    #2*2*400
    network["pool12"] = tf.layers.max_pooling2d(inputs=network["batch_norm12"], pool_size=[2, 2], strides=2)
    
    network["flatten"] = tf.contrib.layers.flatten(network["pool12"])

    network["fc1"] = tf.contrib.layers.fully_connected(inputs=network["flatten"], num_outputs=900, activation_fn=tf.nn.relu)
    network["dropout_fc1"] = tf.layers.dropout(inputs=network["fc1"], rate=0.5)
    network["fc2"] = tf.contrib.layers.fully_connected(inputs=network["dropout_fc1"], num_outputs=200, activation_fn=tf.nn.relu)
    if is_train:
        network["fc3"] = tf.contrib.layers.fully_connected(inputs=network["fc2"], num_outputs=class_num, activation_fn=None)
    else:
        network["fc3"] = tf.contrib.layers.fully_connected(inputs=network["fc2"], num_outputs=class_num, activation_fn=tf.nn.sigmoid)
    network["outputs"] = tf.reshape(network["fc3"], [batch_size, -1, class_num])
    network["outputs"] = tf.transpose(network["outputs"], (1, 0, 2))            # ?

    return network


#
# 读入TFRecord数据并解码：
#
def read_and_decode(fileName):
    # generate a queue with a given file name
    print("reading tfrecords from {}".format(fileName))
    
    filename_queue = tf.train.string_input_producer([fileName])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)     # return the file and the name of file
    
    features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                           features={
                                               'label': tf.VarLenFeature(tf.int64),
                                               'img_raw': tf.FixedLenFeature([],tf.string),
                                           })
    # You can do more image distortion here for training data
    label = tf.cast(features['label'], tf.int64)
    img = tf.decode_raw(features['img_raw'], tf.uint8)  # notice the type of data

    # img = tf.reshape(img, [32,280,1])               # 为何变为[32, 280]? 如何填充0？
    # 每张图片的宽、高和颜色通道。还是高、宽？
    img = tf.reshape(img, [32, 96, 1])               # 为何变为[32, 280]? 如何填充0？

    # 归一化？
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img, label


# 神经网络设置及方法定义：
class SlidingConvolution(object):
    # 初始设置：
    def __init__(self, is_train=False):
        # self.class_num = 5990           # 要识别的分类类别
        # class_num = 标签种类数 + 1，(num_classes - 1)保留用以代表空格标签
        self.class_num = 37             # 分类类别:数字10+字母26，与ocr_char.txt中的数量相同(为何多1？)
        self.character_step = 8         # ？移动的距离，和输入图片宽度、字符宽度共同确定可以识别的最大字符数量，
                                        # 如；（280-32）/8 = 31。过小训练量加大！
        self.character_height = 32      # 转换为280*32固定大小，若文字高度本身很小呢？
        self.character_width = 32       #

        # 训练设置：
        self.train_tfrecords_name = "./training_data_char/captcha.tfrecord"         # 训练数据路径
        self.train_ocrchar_name = "./training_data_char/char_dict.txt"            # 训练数据路径
        self.summary_save_path = "./saving_model_char/"                          # 可以一个目录
        self.summary_steps = 100
        self.save_steps = 10
        self.save_path = "./saving_model_char/"

        # 测试设置:
        self.model_path = "./saving_model_char/sliding_conv.ckpt-740"

        if is_train:
            self.batch_size = 64
            self.with_clip = True       # 采用Gradient Clipping
            self.network = train_network(batch_size=self.batch_size, class_num=self.class_num,
                                   character_height=self.character_height, character_width=self.character_width,
                                   character_step=self.character_step, is_train=True)
        else:
            # 预测：
            current_path = os.path.dirname(os.path.abspath(__file__))

            # 字典对照表
            self.char_dict = self.create_char_dict(
                # os.path.join(current_path, "char_word.txt))
                self.train_ocrchar_name)        # 样本中的所有字符

            self.batch_size = 1
            self.graph = tf.Graph()
            
            # 需了解环境的情况：
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)

            with self.session.as_default():
                with self.graph.as_default():
                    self.network = train_network(batch_size=self.batch_size, class_num=self.class_num,
                                        character_height=self.character_height, character_width=self.character_width,
                                        character_step=self.character_step, is_train=False)

                    # ct_greedy_decoder根据当前序列预测下一个字符，并且取概率最高的作为结果，再此基础上再进行下一次预测。
                    # 输入：
                    #     inputs: 一个3-D Tensor (max_time * batch_size * num_classes),保存着logits(通常是RNN接上一个输出)
                    #     sequence_length: 1-D int32 向量, size为 [batch_size].序列的长度.
                    #     merge_repeated: Default: True.
                    # 输出：
                    #     decoded: decoded[0]是SparseTensor,保存着解码的结果, 分别为索引矩阵、值向量和shape！.
                    self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.network["outputs"], self.network["seq_len"],
                                                                 merge_repeated=True)

                    init = tf.global_variables_initializer()

                    self.session.run(init)
                    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

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
        print("DICT:", char_dict)
        return char_dict

    # 根据训练结果解析对应的字符串：
    def adjust_label(self, result):
        result_str = ""
        # print(self.char_dict)
        print("Result(before):", result, " len=", len(result))
        for x, char in enumerate(result):
            result_str += self.char_dict[char]

        print("Result(after)=[", result_str, "]", " len=", len(result_str))
        return result_str

    # 模型训练：
    def train_model(self):
        # 读取并解码数据：
        train_data, train_label = read_and_decode(self.train_tfrecords_name)

        # train_inputs是Tensor,每个元素为（32,280,1），targets是稀疏张量（SparseTensor），高效存储很多为0的数据。
        # train_inputs是Tensor,每个元素为（32,96,1），targets是稀疏张量（SparseTensor），高效存储很多为0的数据。
        # 每次SHUFFLE生成指定batch_size个的数据：因此train_inputs变为(batch_size, 32, 96, 1), 而train_data为（32,96,1）
        train_inputs, train_targets = tf.train.shuffle_batch([train_data, train_label],
                                                             batch_size=self.batch_size, capacity=50000,  # capacity队列中最大的元素数
                                                             min_after_dequeue=10000)        # 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别
        print("SHUFFLE(train_inputs, train_data): ", train_inputs, train_data)
        global_step = tf.Variable(0, trainable=False)
        
        # 学习率设置：
        # learning_rate = tf.train.exponential_decay(1e-3,
        learning_rate = tf.train.exponential_decay(0.001,
                                                   global_step,
                                                   # 10000,
                                                   500,            # 每1000轮后*0.9
                                                   0.95,
                                                   staircase=True)
        # SparseTensor:
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # 损失函数：处理输出标签预测值network["outputs"]和真实标签之间的损失
        # 经过RNN后输出的标签预测值为3D浮点Tensor，默认形状为max_time * batch_size * num_classes，max_time？
        # sequence_length: 1-D int32 vector, 大小为[batch_size]，vector中的每个值表示序列的长度.
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, inputs=self.network["outputs"], sequence_length=self.network["seq_len"]))
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
        decoded, log_prob = tf.nn.ctc_greedy_decoder(self.network["outputs"], self.network["seq_len"], merge_repeated=True)
        acc = 1 - tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        tf.summary.scalar("accuracy", acc)
        merge_summary = tf.summary.merge_all()

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)        # 作用？
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)        # 作用？
        print(gpu_options)
        init = tf.global_variables_initializer()
        
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(init)
            threads = tf.train.start_queue_runners(sess=session)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            if LOAD_MODEL:
                saver.restore(session, self.model_path)                     # 恢复保存的模型，继续训练！
                print("Restore training model: ", self.model_path)

            summary_writer = tf.summary.FileWriter(self.summary_save_path, session.graph)
            
            while True:
                step_input, step_target, steps, lr = session.run(
                    [train_inputs, train_targets, global_step, learning_rate])
                feed = {self.network["inputs"]: step_input, targets: step_target}
                # print("FEED:", feed)

                batch_acc, batch_loss, _ = session.run([acc, loss, optimizer_op], feed)

                print("Step is: {}, batch loss is: {} learningrate is {}， acc is {}".format(steps, batch_loss, lr, batch_acc))
                
                if steps > 0 and steps % self.summary_steps == 0:
                    _, batch_summarys = session.run([optimizer_op, merge_summary], feed)
                    summary_writer.add_summary(batch_summarys, steps)

                if steps > 0 and steps % self.save_steps == 0:
                    save_path = saver.save(session, os.path.join(self.save_path, "sliding_conv.ckpt"), global_step=steps)
                    print(save_path)

    # 测试：
    def predict_model(self, input_data):
        if input_data.ndim == 3:
            print("ndim is 3 ......")
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)

        height, width = input_data.shape
        print("Photo Size: ", height, width, input_data.shape)
        ratio = height / 32.0

        # input_data = cv2.resize(input_data, (32, int(width / ratio)))
        # 1, 32, 96, 1
        # input_data = cv2.resize(input_data, (32, 96))
        input_data = cv2.resize(input_data, (96, 32))           # !!!!!!!!!!!!
        print("Photo Size1: ", input_data.shape)

        # input_data.save('./Label_convert.jpg')  # 存下图片

        # input_data = input_data.reshape((1, 32, int(width / ratio), 1))
        input_data = input_data.reshape((1, 32, 96, 1))
        scaled_data = np.asarray(input_data / np.float32(255) - np.float32(0.5))

        with self.session.as_default():
            with self.graph.as_default():
                feed = {self.network["inputs"]: scaled_data}

                # outputs为ndarray，其长度可变，随预测文字的长度而变化。decoded为SparseTensorValue:
                # 应该用outputs值！ decoded是预测的下一个值！
                outputs, decoded = self.session.run([self.network["outputs"], self.decoded[0]], feed)

                predict = []
                for k in range(outputs.shape[0]):   # 很多字母的概率都很大，很难区分出来！
                    # 分别找到概率最大和最小的位置，从而可找到相应的字母: 比实际位置+1了？
                    pos = list(map(list(outputs[k][0]).index, heapq.nlargest(2, list(outputs[k][0]))))
                    # print(pos)
                    predict.append(pos[0])
                    # print(np.max(outputs[k][0]), np.min(outputs[k][0]))
                    if k == 0:
                        # print(outputs[k][0][0:37])
                        print("OUTPUT SHAPE:", outputs.shape)
                    # for j in range(outputs[k][0].size):
                        # if outputs[k][0][j] > 0.8:
                            # print(j, outputs[k][0][j])

                print("OUTPUTS:", type(outputs), outputs.size, outputs.shape)
                print("PREDICT:", predict)
                print("DECODED:", type(decoded), decoded)                    # decoded[0]为Sparse值的位置, [1]为值，[2]为shape

                result_str = self.adjust_label(predict)

                result_str = self.adjust_label(decoded[1])
                return result_str


def main():
    # if tf.app.flags.FLAGS.mode == "train":
    if TRAINING:
        print("Training ....")
        sc = SlidingConvolution(is_train=True)
        sc.train_model()
    else:
        print("Predicting ....")
        image = cv2.imread("./make_tfrecords/images/captcha_png/4TV0_1530629973.png", 1)
        # image = cv2.imread("./make_tfrecords/images/captcha_png/0FJP_1530629934.png", 1)
        # image = cv2.imread("./char6.png", 1)

        # image = cv2.imread("./make_tfrecords/images/captcha_png/O3K5_1530629902.png", 1)

        cv2.namedWindow("Image")
        cv2.imshow("Image", image)
        cv2.waitKey(2)


        sc = SlidingConvolution(is_train=False)
        # print(sc.adjust_label([12, 10, 1, 5]))
        sc.predict_model(image)
        sc.predict_model(255-image)


if __name__ == '__main__':
    # tf.app.run()       # main() takes 0 positional arguments but 1 was given
    main()

