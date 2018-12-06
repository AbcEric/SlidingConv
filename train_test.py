# Attention：
# 1.lable.txt中标签的长度不能过长：tf.nn.ctc_loss会遇到问题，最大长度由(输入图片宽度-字符宽度)/移动距离决定;
#   太长也会导致ctc_loss出现"No valid path found"
# 2.根据样本调整参数，包括：样本的类别总数（过大会导致部分字符的权重极高，很难判断是哪一个？），图片宽度，字符的长宽等；

# Q:
# 1.CUDA的使用？GPU型号？TF中如何设置？
# 2.滑动窗口用于目标检测：对每个滑动区域预测目标出现的概率，计算成本高！窗口过大会影响性能，过小计算量大。
# 3.如何多线程？
# 4.leariningrate先快后慢？
# 5.CTC？

# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
from model import network
from data_gen import read_and_decode
import tensorflow as tf
import heapq

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
#tf.app.flags.DEFINE_string('mode', 'test', 'train or test')

# 神经网络设置及方法定义：
class Sliding_Convolution(object):
    # 初始设置：
    def __init__(self, is_train=False):
        #self.class_num = 5990           # 要识别的分类类别
        self.class_num = 80             # 分类类别:数字、字母和常见符号
        self.character_step = 8         # ？移动的距离，和输入图片宽度、字符宽度共同确定可以识别的最大字符数量，
                                        # 如；（280-32）/8 = 31。过小训练量加大！
        self.character_height = 32      # 转换为280*32固定大小，若文字高度本身很小呢？
        self.character_width = 32       #

        # 训练设置：
        self.train_tfrecords_name = "./make_tfrecords/data.tfrecords"       # 训练数据路径
        #self.train_tfrecords_name = "./make_tfrecords/ocr.tfrecords"       # 训练数据路径
        self.summary_save_path = "./summary/"
        self.summary_steps = 100
        self.save_steps = 10
        self.save_path = "./save/"

        # 测试设置:
        self.model_path = "./save/sliding_conv.ckpt-60"

        # 训练
        if is_train:
            self.batch_size = 16
            self.with_clip = True       # 采用Gradient Clipping
            self.network = network(batch_size=self.batch_size, class_num=self.class_num,
                                   character_height=self.character_height, character_width=self.character_width,
                                   character_step=self.character_step, is_train=True)
        # 预测：
        else:
            current_path = os.path.dirname(os.path.abspath(__file__))

            # 字典对照表
            self.char_dict = self.create_char_dict(
#                os.path.join(current_path, "char_std_5990.txt"))
                os.path.join("./make_tfrecords/images", "char_word.txt"))

            self.batch_size = 1         # 单张图片预测
            self.graph = tf.Graph()

            # 核实！
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)

            with self.session.as_default():
                with self.graph.as_default():
                    self.network = network(batch_size=self.batch_size, class_num=self.class_num,
                                        character_height=self.character_height, character_width=self.character_width,
                                        character_step=self.character_step, is_train=False)

                    # ct_greedy_decoder根据当前序列预测下一个字符，并且取概率最高的作为结果，再此基础上再进行下一次预测。
                    # 输入：
                    #     inputs: 一个3-D Tensor (max_time * batch_size * num_classes),保存着logits(通常是RNN接上一个线性神经元的输出)
                    #     sequence_length: 1-D int32 向量, size为 [batch_size].序列的长度.和用在dynamic_rnn中的sequence_length一致,使用来表示rnn的哪些输出不是pad的.
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
        return

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
        print("Result(before):", result)
        for x, char in enumerate(result):
            result_str += self.char_dict[char]

        return result_str

    # 模型训练：
    def train_net(self):
        # 读取并解码数据：
        train_data, train_label = read_and_decode(self.train_tfrecords_name)
        #print("DECODE LABEL: ", train_label)

        # train_inputs是Tensor,每个元素为（32,280,1），targets是稀疏张量（SparseTensor），高效存储很多为0的数据。
        # 每次生成指定batch_size个的数据：
        train_inputs, train_targets = tf.train.shuffle_batch([train_data, train_label],
                                                             batch_size=self.batch_size, capacity=2000,
                                                             min_after_dequeue=1000)

        global_step = tf.Variable(0, trainable=False)

        # 学习率设置：
        learning_rate = tf.train.exponential_decay(1e-3,
                                                   global_step,
                                                   10000,
                                                   0.9,
                                                   staircase=True)

        # 标签目标值：训练时带入真实值
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # 损失函数：处理输出标签预测值network["outputs"]和真实标签之间的损失
        # 经过RNN后输出的标签预测值为3D浮点Tensor，默认形状为max_time * batch_size * num_classes，max_time？
        # sequence_length: 1-D int32 vector, 大小为[batch_size]，vector中的每个值表示序列的长度.
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, inputs=self.network["outputs"], sequence_length=self.network["seq_len"]))
        tf.summary.scalar("loss", loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)             # 作用？

        if update_ops:
            with tf.control_dependencies([tf.group(*update_ops)]):
                if self.with_clip == True:                                  # 采用Gradient Clipping
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
            if self.with_clip == True:
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

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)            # 作用？
        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(init)
            threads = tf.train.start_queue_runners(sess=session)        # 避免无数据挂起？
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            saver.restore(session, self.model_path)                     # 恢复保存的模型，继续训练！
            summary_writer = tf.summary.FileWriter(self.summary_save_path, session.graph)

            while True:
                # 单步的输入和标签：step_target运行run后生成的SparseTensorValue数据，而train_target是前面定义的图(SparseTensor)。
                # shuffle_batch()会生成一批训练数据train_inputs和train_targets
                step_input, step_target, steps, lr = session.run(
                    [train_inputs, train_targets, global_step, learning_rate])
                #print("TARGETS: ", type(step_target), type(train_targets))

                # inputs的定义为：[batch_size, 32, None, 1]
                feed = {self.network["inputs"]: step_input, targets: step_target}

                # 传入数据后运行：
                batch_acc, batch_loss, _ = session.run([acc, loss, optimizer_op], feed)

                print("step is: {}, batch loss is: {} learningrate is {} acc is {}".format(steps, batch_loss, lr, batch_acc))
                
                if steps > 0 and steps % self.summary_steps == 0:
                    _, batch_summarys = session.run([optimizer_op, merge_summary], feed)
                    summary_writer.add_summary(batch_summarys, steps)

                if steps > 0 and steps % self.save_steps == 0:
                    save_path = saver.save(session, os.path.join(self.save_path,"sliding_conv.ckpt"), global_step=steps)
                    print(save_path)

    # 预测：
    def test_net(self, input_data):
        if input_data.ndim == 3:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)

        height, width = input_data.shape
        print("Photo Size: ", height, width)

        ratio = height / 32.0

        # 数据做归一化处理：
        input_data = cv2.resize(input_data, (32, int(width / ratio)))
        input_data = input_data.reshape((1, 32, int(width / ratio), 1))
        scaled_data = np.asarray(input_data / np.float32(255) - np.float32(0.5))
        print("Data Shape: ", scaled_data.shape)

        with self.session.as_default():
            with self.graph.as_default():
                # 比较一下scaled_data和训练时的step_input格式是否一致？
                feed = {self.network["inputs"]: scaled_data}

                # network["outputs"]?
                # outputs为ndarray，其长度可变，随预测文字的长度而变化。decoded为SparseTensorValue:
                # 应该用outputs值！ decoded是预测的下一个值！
                outputs, decoded = self.session.run([self.network["outputs"], self.decoded[0]], feed)

                for k in range(outputs.shape[0]):
                    print(list(map(list(outputs[k][0]).index, heapq.nsmallest(2, list(outputs[k][0])))))
                    print(np.max(outputs[k][0]), np.min(outputs[k][0]))
                    for j in range(outputs[k][0].size):
                        if outputs[k][0][j] > 0.8:
                            print(j, outputs[k][0][j])

                print("OUTPUTS:", type(outputs), outputs.size, outputs.shape)
                print("DECODED:", type(decoded), decoded[1])                    # decoded[0]为Sparse值的位置, [1]为值，[2]为shape
                result_str = self.adjust_label(decoded[1])
                return result_str

def main():
    '''
    tfrecord_name = "data.tfrecords"
    print("Writing ....")
    #data_to_tfrecord(["./make_tfrecords/images/output/7afd99ab6357cd274093011564315e93.jpg", \
    #                  "./make_tfrecords/images/output/e13c79f8622c2ed0e18da98c2281f39a.jpg"], [['9','3','7','2','20', '1','1','1','5'], ['1', '2', '7']], tfrecord_name)

    print("Reading ....")
    img, label = read_and_decode(tfrecord_name)
    print("SIZE: ", type(label))
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=2, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.global_variables_initializer()

    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    with tf.Session() as ss:
        ss.run(init)
        tf.train.start_queue_runners(sess=ss)
        print(ss.run(label))
        print(ss.run(img))

    exit(0)
    '''
    print(tf.app.flags.FLAGS.mode)
    # train
    if tf.app.flags.FLAGS.mode == "train":
        sc = Sliding_Convolution(is_train=True)
        sc.train_net()
    # test
    else:
        image = cv2.imread("./make_tfrecords/images/output/d2811ec0d53f6a8771f3e8c8fc641423.jpg", 1)
        #image = cv2.imread("./make_tfrecords/images/output/7afd99ab6357cd274093011564315e93.jpg", 1)
        #image = cv2.imread("./make_tfrecords/images/output/2b153238a1a838a5775bb2704d791f8c.jpg", 1)

        cv2.namedWindow("Image")
        cv2.imshow("Image", image)
        cv2.waitKey(3)

        sc = Sliding_Convolution(is_train=False)
        # print(sc.adjust_label([12, 10, 1, 5]))

        result_str = sc.test_net(image)
        print("result: ", result_str)

if __name__ == '__main__':
#    tf.app.run()
    main()