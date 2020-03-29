import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np


def lenet(num_classes, lambd):
    network = Sequential([
        layers.Conv2D(6, kernel_size=3, strides=1, padding='SAME', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(l=lambd)),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Conv2D(16, kernel_size=3, strides=1, padding='SAME', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(l=lambd)),
        layers.MaxPooling2D(pool_size=2, strides=2),

        layers.Flatten(),

        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=lambd)),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=lambd)),
        layers.Dense(num_classes, activation=None)
    ])
    return network


class BasicBlock(layers.Layer):
    # 残差模块
    def __init__(self, filter_num, lambd, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积单元
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=lambd))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二个卷积单元
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=lambd))
        self.bn2 = layers.BatchNormalization()

        if stride != 1:  # 通过1x1卷积完成shape匹配
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:  # shape匹配，直接短接
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        # [b, h, w, c]，通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        # 通过identity模块
        identity = self.downsample(inputs)
        # 2条路径输出直接相加
        output = layers.add([out, identity])
        output = tf.nn.relu(output)  # 激活函数

        return output


class ResNet(keras.Model):
    # 通用的ResNet实现类
    def __init__(self, layer_dims, num_classes, lambd, summary_writer):
        super(ResNet, self).__init__()
        self.summary_writer = summary_writer
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1),
                                              kernel_regularizer=tf.keras.regularizers.l2(l=lambd)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                # layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 堆叠4个Block，每个block包含了多个BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64, layer_dims[0], lambd=lambd)
        self.layer2 = self.build_resblock(128, layer_dims[1], lambd=lambd, stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], lambd=lambd, stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], lambd=lambd, stride=2)
        self.ConvT1 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='valid')
        self.ConvT2 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='valid')
        self.ConvT3 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='valid')
        self.ConvT4 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(4, 4), padding='same')
        self.ConvT5 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(8, 8), padding='same')
        # 通过Pooling层将高宽降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # 通过根网络
        with self.summary_writer.as_default():
            tf.summary.image('Training data', inputs, max_outputs=5, step=0)
            x = self.stem(inputs)
            x_map = self.ConvT1(x)
            tf.summary.image('conv1', x_map, max_outputs=5, step=0)
            # 一次通过4个模块
            x = self.layer1(x)
            x_map = self.ConvT2(x)
            tf.summary.image('conv2', x_map, max_outputs=5, step=0)
            x = self.layer2(x)
            x_map = self.ConvT3(x)
            tf.summary.image('conv3', x_map, max_outputs=5, step=0)
            x = self.layer3(x)
            x_map = self.ConvT4(x)
            tf.summary.image('conv4', x_map, max_outputs=5, step=0)
            x = self.layer4(x)
            x_map = self.ConvT5(x)
            tf.summary.image('conv5', x_map, max_outputs=5, step=0)
            # 通过池化层
            x = self.avgpool(x)
            # 通过全连接层
            x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, lambd, stride=1):
        # 辅助函数，堆叠filter_num个BasicBlock
        res_blocks = Sequential()
        # 只有第一个BasicBlock的步长可能不为1，实现下采样
        res_blocks.add(BasicBlock(filter_num, lambd, stride))

        for _ in range(1, blocks):  # 其他BasicBlock步长都为1
            res_blocks.add(BasicBlock(filter_num, lambd=lambd, stride=1))

        return res_blocks


def resnet18(num_classes, lambd, summary_writer):
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([2, 2, 2, 2], num_classes, lambd, summary_writer)


def resnet34(num_classes, lambd, summary_writer):
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([3, 4, 6, 3], num_classes, lambd, summary_writer)
