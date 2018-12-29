# coding: utf-8


import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import time


CONTENT_IMG = 'cas.jpg'  # 内容图像
STYLE_IMG = 'style2.jpg'  # 风格图像
OUTPUT_DIR = 'output/'  # 输出路径 
IMAGE_W = 800  # 图像宽度
IMAGE_H = 600  # 图像高度
COLOR_C = 3  # 3 通道
NOISE_RATIO = 0.7  # 生成噪声图片
BETA = 5  # 内容损失的权重
ALPHA = 100  # 风格损失的权重
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'  # VGG imagenet 权值 
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))  # 通道颜色均值
STYLE_LAYERS = [('conv1_1', 0.5), ('conv2_1', 1.0), ('conv3_1', 1.5), ('conv4_1', 3.0)]  # 定义用于风格损失计算的层，每一层有一个权值，更深的层对应的权值更高
CONTENT_LAYER = 'conv4_2'  # 定义用于内容损失计算的层


if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)


def the_current_time():
    """
    打印当前时间
    """
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))


def load_vgg_model(path):
    """
    加载 VGG 模型
    """

    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        """
        加载权重 W, b
        """
        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        定义卷积层
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)  # W 为常量
        b = tf.constant(np.reshape(b, (b.size)))  # b 为常量
        return tf.nn.relu(tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)

    def _avgpool(prev_layer):
        """
        定义池化层
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    graph = {}
    graph['input']    = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')  # 输入为 Variable，反向调参，调整的是输入，W 和 b 为常量在训练过程中不变
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph


def content_loss_func(sess, model):
    """
    内容损失函数
    """

    def _content_loss(p, x):
        N = p.shape[3]  # feature_map 的数量（通道数）
        M = p.shape[1] * p.shape[2]  # feature_map 的尺寸
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))

    return _content_loss(sess.run(model[CONTENT_LAYER]), model[CONTENT_LAYER])


def style_loss_func(sess, model):
    """
    风格损失函数
    """

    def _gram_matrix(F, N, M):
        Ft = tf.reshape(F, (M, N))  # 把 F（卷积层的输出） 转成 M * N 的矩阵，M 为 feature_map 尺寸，N 为 feature_map 数量 
        return tf.matmul(tf.transpose(Ft), Ft)  # (feature_map 数量 * feature_map 尺寸) * (feature_map 尺寸 * feature_map 数量)

    def _style_loss(a, x):
        N = a.shape[3]  # feature_map 的数量
        M = a.shape[1] * a.shape[2]  # feature_map 的尺寸
        A = _gram_matrix(a, N, M)
        G = _gram_matrix(x, N, M)
        return (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(G - A, 2))

    return sum([_style_loss(sess.run(model[layer_name]), model[layer_name]) * w for layer_name, w in STYLE_LAYERS])


def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    """
    基于内容图像产生一张随机图片
    基于内容图像生成随机图片的目的是为了加速训练
    """
    # 从一个均匀分布中随机抽样 [-20, 20)
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, COLOR_C)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


def load_image(path):
    """
    加载一张图片
    """
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (IMAGE_H, IMAGE_W))
    image = np.reshape(image, ((1, ) + image.shape))  # 扩充了 1 维，3 维变 4 维
    image = image - MEAN_VALUES  # 减去通道颜色均值
    return image


def save_image(path, image):
    """
    存储一张图像
    """
    image = image + MEAN_VALUES  # 加上通道颜色均值
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


with tf.Session() as sess:
    the_current_time()  # 打印当前时间
    content_image = load_image(CONTENT_IMG)  # 加载内容图像
    style_image = load_image(STYLE_IMG)  # 加载风格图像
    model = load_vgg_model(VGG_MODEL)  # 加载模型

    input_image = generate_noise_image(content_image)  # 生成一张随机图片
    sess.run(tf.global_variables_initializer())  # 网络参数初始化

    sess.run(model['input'].assign(content_image))  # assigns a new value to the variable(内容图像)，model['input']: tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')
    content_loss = content_loss_func(sess, model)  # 计算内容损失

    sess.run(model['input'].assign(style_image))  # assigns a new value to the variable(风格图像)，model['input']: tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')
    style_loss = style_loss_func(sess, model)  # 计算风格损失

    total_loss = BETA * content_loss + ALPHA * style_loss  # 加权计算总体损失
    optimizer = tf.train.AdamOptimizer(2.0)  # 定义优化器，使用 adam 优化算法
    train = optimizer.minimize(total_loss)

    sess.run(tf.global_variables_initializer())  # 网络参数初始化
    sess.run(model['input'].assign(input_image))  # assign a new value to the variable(随机生成的图片)

    ITERATIONS = 2000  # 定义最大的迭代次数
    for i in range(ITERATIONS):
        sess.run(train)
        if i % 100 == 0:
        	output_image = sess.run(model['input'])  # 获取当前网络的输入
        	the_current_time()  # 打印当前时间
        	print('Iteration: {}, Cost: {}'.format(i, sess.run(total_loss)))
        	save_image(os.path.join(OUTPUT_DIR, 'output_{}_{}.jpg'.format(STYLE_IMG.split('.')[0], i)), output_image)
    save_image(os.path.join(OUTPUT_DIR, 'output_{}_{}.jpg'.format(STYLE_IMG.split('.')[0], ITERATIONS)), output_image)