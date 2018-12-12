#
# 验证码图片处理：降噪，二值化，切分，格式转换等
#

'''
修改历史：
1. 下一步处理为常用功能函数：
2. 需要改进输入数据的处理，不再采用mnist方式，自己写，或者tf.data.Dataset,参见：https://github.com/SymphonyPy/Valified_Code_Classify
3. 图片能否自动改变分辨率到120x40？

'''
from skimage import io, color, filters, util, data_dir
import os
import matplotlib.pyplot as plt
from ctypes import create_string_buffer
from PIL import Image
import struct
import datetime

# 八邻域降噪：对传入二值化后的图片进行降噪
def depoint(img):
    pixdata = img
    w, h = img.shape

    # 去除图像边缘噪点
    for x in range(w):
        pixdata[x, h - 1] = 255
        pixdata[x, 0] = 255
    for x in range(h):
        pixdata[w - 1, x] = 255
        pixdata[0, x] = 255

    # 对一个像素点周八个点进行搜寻并统计白点数量
    i = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            count = 0
            if pixdata[x, y] == 0:
                if pixdata[x, y - 1] == 255:  # 上
                    count = count + 1
                if pixdata[x, y + 1] == 255:  # 下
                    count = count + 1
                if pixdata[x - 1, y] == 255:  # 左
                    count = count + 1
                if pixdata[x + 1, y] == 255:  # 右
                    count = count + 1
                if pixdata[x - 1, y - 1] == 255:  # 左上
                    count = count + 1
                if pixdata[x - 1, y + 1] == 255:  # 左下
                    count = count + 1
                if pixdata[x + 1, y - 1] == 255:  # 右上
                    count = count + 1
                if pixdata[x + 1, y + 1] == 255:  # 右下
                    count = count + 1
                if count > 5:  # 如果周围白点数量大于5则判定为噪点
                    i = i + 1  # 统计该次降噪处理了多少个噪点
                    pixdata[x, y] = 255
    return i

# 图片预处理：转为灰度，二值化
def ImagePretreatment(file_img):
    path = os.getcwd()
    #path_img = os.path.join(path, 'img_data')
    fullfile = os.path.join(path, file_img)

    # scikit-image 图像处理模块： pip install scikit-image
    image = io.imread(fullfile)  # 为图像的color？

    # 降噪：
    depoint(image)
    depoint(image)

    image = color.rgb2gray(image)  # 灰度, 转换结果在[0,1]之间

    # print(image)
    thresh = filters.threshold_otsu(image)  # 获得阈值
    # image = (image <= thresh) * 1.0  # 根据阈值进行分割

    image = (image >= thresh) * 255  # 二值化: 像素值大于阈值的变为255，否则变为0
    # image = (image <= thresh) * 255         # 二值化: 像素值大于阈值的变为255，否则变为0


    return image

# 记录日志：
def writeLog(logfile, log):
    with open(logfile, 'a+') as f:
        t = datetime.datetime.now()
        line = '\n[%(year)04d-%(mon)02d-%(day)02d] - [%(hour)02d:%(minute)02d:%(second)02d]\n' % {"year": t.year, "mon": t.month, "day": t.day, "hour": t.hour, "minute": t.minute, "second": t.second}
        f.write(line)
        f.write(log + "\n")
        f.close()

        return 0


# 图片转换：将验证码图片进行降噪等，切分为4张小图片
def VerifycodeImageConvert():
    path = os.getcwd()
    path_img = os.path.join(path, 'img_data')
    path_conv = os.path.join(path, 'img_conv')

    img_file = os.listdir(path_img)
    num = 0     # 切割后的图片顺序编号

    for img in img_file:
        file_img = os.path.join(path_img, img)

        # 预处理：
        image = ImagePretreatment(file_img)
        # 降噪：
        depoint(image)

        # 分割后保存
        j = 0

        while j < 4:
            savefile = "%(num)04d.jpg" % {'num': num}
            file_save = os.path.join(path_conv, savefile)
            io.imsave(file_save, image[0:40, j * 30:(j + 1) * 30])
            j += 1
            num += 1

    return 0

# 读取保持的分类结果：[文件名，分类名]
def loadNumber(lablefile):
    numberList = []

    with open(lablefile, "r") as f:
        total = 0
        for line in f.readlines():
            item = line[:-1].split(' ')
            #print(item[0], item[1])
            if item[1] != '':
                numberList.append(item)
                total += 1

    return numberList

# 图片数据转换为mnist格式：输入：[图像目录文件名，生成文件名]
def ImageConvert2Mnist(imgpath, outputimgdata, labelfile, outputlabel):
    # 转换后的图片路径
    path = os.getcwd()
    path_conv = os.path.join(path, imgpath)

    # 读取文件列表：控制只有*.jpg文件？
    filelist = os.listdir(path_conv)
    imgfile = os.path.join(path_conv, filelist[0])
    #print(imgfile)
    #imgfile = os.path.join(path_conv, numberList[0][0])

    # 读取标记的结果：
    numberList = loadNumber(labelfile)
    #print(imgfile)

    # 图片文件头：magic, 图像总数，rows，columns
    imgindex = 0
    imgmagic = 2051                         # 固定值
    imgnum = len(numberList)                # 图片是否写入以label文件为准
    image = Image.open(imgfile)         # 读取一个文件来获得图片的相关信息
    rows = image.size[1]                    # 高度
    columns = image.size[0]                 # 宽度
    imgbuf = create_string_buffer(16 + rows*columns*imgnum)

    # 写入图片头部：
    struct.pack_into('>IIII', imgbuf, imgindex, imgmagic, imgnum, rows, columns)
    print(imgmagic, " ", imgnum, " ", rows, " ", columns)

    imgindex = struct.calcsize('>IIII')            # 头部长度

    # 标记文件头：magic, 图像总数，rows，columns
    labelindex = 0
    labelmagic = 2049                        # 固定值
    labelnum = len(numberList)            # 图片是否写入以label文件为准

    # 标记类型是unsigned byte,标签为0-9（1个字节）
    #print("labelnum:", labelnum)
    labelbuf = create_string_buffer(8 + labelnum)

    # 写入头部：
    struct.pack_into('>II', labelbuf, labelindex, labelmagic, labelnum)
    print(labelmagic, " ", labelnum)

    labelindex = struct.calcsize('>II')            # 头部长度

    # 写入图片和标签数据
    for item in numberList:
        #print(item)
        #imgfile = os.path.join(path_conv, img)
        imgfile = os.path.join(path_conv, item[0])

        # image是numpy.ndarray类型, 每个元素是0-255的整数
        image = io.imread(imgfile)

        for x in range(rows):
            for y in range(columns):
                # print(image[y][x])
                struct.pack_into('>B', imgbuf, imgindex, image[x][y])
                imgindex += struct.calcsize('>B')

        struct.pack_into('>B', labelbuf, labelindex, int(item[1]))
        labelindex += struct.calcsize('>B')

    # 保持结果到文件：
    fp = open(outputimgdata, "wb")  # 二进制追加
    fp.write(imgbuf)
    fp.close()

    # 写入标签数据
    fp = open(outputlabel,"wb")
    fp.write(labelbuf)
    fp.close()

    return(0)


# 解析minist数据
# python没有没有处理字节的数据类型，采用struct模块来解决str和其他二进制数据类型的转换。
# pack可以将任何数据（如图像信息）转换为字符串！
def ReadMinistImage(filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    # images为图片个数：
    # '>'表示字节顺序是big-endian(大端规则)，也就是网络序，
    # 'I'表示4字节无符号整数uint32，如：\x00\x00\x00\x01来表示数字1（同理uint8为1字节？） 'H'为2字节无符号整形
    # 'IIII'则总字节数为4x4=16. 如果要读取一个字节用'>B'（如：\x01来表示数字1，必须在0-255之间）
    # index开始读取的位置
    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    print("解析结果：")
    print(magic, " ", images, " ", rows, " ", columns)

    index += struct.calcsize('>IIII')

    for i in range(images):
        # 代表为灰度图
        image = Image.new('L', (columns, rows))
        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

    image.save(str(i) + '.jpg')

        #image.save('minist/' + str(i) + '.jpg')

    print('Just save the last images: ', i, '.jpg. Total imgnumber = ',i+1)

# 解析minist标签数据
def ReadMinistLabel(filename, savefilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')

    labelArr = [0] * labels

    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
        save = open(savefilename, 'w')

        save.write(','.join(map(lambda x: str(x), labelArr)))

        save.write('\n')

    save.close()

    print('Save labels success: ', savefilename)


if __name__ == '__main__':
    # 切分图片：
    VerifycodeImageConvert()
    exit(0)
    # 转换切分后的图片数据：转换为mnist格式
    # 先用Bbox-Label-Tool工具对切分后的数据打标签
    ImageConvert2Mnist('img_conv', 'train-images.idx3-ubyte', 'train.txt', 'train-labels.idx1-ubyte')
    #ImageConvert2Mnist('img_conv', 't10k-images.idx3-ubyte', 'verify.txt', 't10k-labels.idx1-ubyte')

    ReadMinistImage('train-images.idx3-ubyte')
    #ReadMinistImage('t10k-images.idx3-ubyte')

    # 安装7-zip软件进行压缩，格式gzip。压缩后的文件名：train-labels-idx1-ubyte.gz!!!
    ReadMinistLabel('train-labels.idx1-ubyte', 'label_out.txt')
    #ReadMinistLabel('t10k-labels.idx1-ubyte', 'label_out.txt')

    # Minist:
    #read_ministimage('t10k-images.idx3-ubyte')
    #read_ministlabel('t10k-labels.idx1-ubyte', 'label_out.txt')
