"""
DenseNet121 implementation in TF2 
Date: Dec 2, 20
Author: YIN Chao
"""

import os
import time
import random

from PIL import Image

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Input, GlobalAveragePooling2D, Add, \
    Dense, Activation, Flatten, \
    BatchNormalization, Conv2D, MaxPooling2D,\
    AveragePooling2D, Concatenate, ReLU

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print(tf.__version__)
np.random.seed(123)
tf.random.set_seed(123)


# define a model
class DenseNet121(Model):
    def __init__(self, num_classes=10, num_channels=64, growth_rate=32, num_convs_in_dense_blocks=[6, 12, 24,16], **kwargs):
        super(DenseNet121, self).__init__(**kwargs)
        self.num_classes = num_classes # cls categories
        self.num_channels = num_channels # 
        self.growth_rate = growth_rate
        self.num_convs_in_dense_blocks = num_convs_in_dense_blocks

    # input_shape is (B,D1,...)
    def build(self, input_shape):
        self.conv1 = Conv2D(64, input_shape=input_shape,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.pool1 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same'
                                  )
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.global_avg = GlobalAveragePooling2D()
        self.fc1 = Dense(100, activation='relu')
        self.fc2 = Dense(self.num_classes, activation='softmax')

    def call(self, x, training=None):
        h = self.relu1(self.bn1(self.conv1(x), training))
        h = self.pool1(h)

        # dense blocks and transition layers
        current_num_channels = self.num_channels
        for i, num_convs in enumerate(self.num_convs_in_dense_blocks):
            h = self.__dense_block(h, num_convs, self.growth_rate)
            current_num_channels += num_convs * self.growth_rate
            if i != len(self.num_convs_in_dense_blocks)-1:
                current_num_channels //= 2
                h = self.__transition_block(h, current_num_channels)

        h = self.relu2(self.bn2(h, training))
        h = self.global_avg(h)
        h = self.fc1(h)
        y = self.fc2(h)
        return y

    # intuitively, one dense block will conduct convolution densely (no spatial size change but channels increase greatly)--(B,H,W,D) --> (B,H,W,num_channels*num_of_blocks+D)
    def __dense_block(self, x, num_of_blocks, num_channels):
        # Note: num_channels also calls the growth rate
        for _ in range(num_of_blocks):
            h = conv_block(num_channels)(x)
            x = Concatenate()([x, h])
        return x

    # 1x1 conv to manage channel size and then down-sample by 2
    def __transition_block(self, x, num_channels):
        blk = tf.keras.Sequential([
            BatchNormalization(),
            ReLU(),
            Conv2D(num_channels, kernel_size=1),
            AveragePooling2D(pool_size=2, strides=2)
        ])

        y = blk(x)
        return y

    # overcome the bug show `multiple for tensor shape` for model.summary() in tf 2.1/2;
    # tf2.3 fixed this bug already
    # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model

    def summary2(self, inputs):
        x = Input(shape=inputs.shape[1:])
        self.build(inputs.shape)
        y = self.call(x)
        model = Model(inputs=[x], outputs=[y])
        return model.summary()


def conv_block(num_channels):
    blk = tf.keras.Sequential([
        BatchNormalization(),
        ReLU(),
        Conv2D(128, kernel_size=1, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(num_channels, kernel_size=3, padding='same'),
    ])

    return blk


if __name__ == "__main__":
    x = np.random.uniform(size=(2, 224, 224, 3))
    # model = DenseNet121(num_convs_in_dense_blocks=[4,4,4,4])
    # 121 layers = 1 (first conv)+ (6+12+24+16)*2(four dense block, 2 is 2 conv) + 3(3 transition layer)+ + 1(fc layers)
    model = DenseNet121(num_convs_in_dense_blocks=[6,12,24,16])
    y = model(x)
    print(y)
    print(model.summary())
    # use my own summary method
    print(model.summary2(x))
    print('success!')
