"""
googlenet implementation in TF2 
Date: Dec 14, 20
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
AveragePooling2D, Layer, ReLU, \
GlobalAveragePooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print(tf.__version__)
np.random.seed(123)
tf.random.set_seed(123)



class ConvBNRelu(Model):
    
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])
        
        
    def call(self, x, training=None):
        
        x = self.model(x, training=training)
        return x 
    
class InceptionBlock(Model):
    
    def __init__(self, filters, strides=1):
        super(InceptionBlock, self).__init__()
        
        self.filters = filters
        self.strides = strides
        
        self.conv1 = ConvBNRelu(filters, strides=strides)
        self.conv2 = ConvBNRelu(filters, kernel_size=3, strides=strides)
        self.conv3_1 = ConvBNRelu(filters, kernel_size=3, strides=strides)
        self.conv3_2 = ConvBNRelu(filters, kernel_size=3, strides=1)
        
        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(filters, strides=strides)
        
        
    def call(self, x, training=None):
        
        
        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)
                
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
                
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)
        
        # concat along axis=channel
        x = tf.concat([x1, x2, x3_2, x4], axis=3)
        
        return x

class InceptionBlock(Model):
    """
    residual block
    """
    # if use_1x1conv=True, strides should be 2 since making sure short's ouput has same shapes as main stream
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(
            filters=num_channels,
            kernel_size=3,
            padding='same',
            strides =strides,
        )
        self.conv2 = Conv2D(
            filters=num_channels,
            kernel_size=3,
            padding='same',
        )
        if use_1x1conv:
            self.conv3 = Conv2D(
                filters=num_channels,
                kernel_size=1,
                padding='same',
                strides =strides,
            )
        else:
            self.conv3= None
        self.relu1= ReLU()
        self.bn1=BatchNormalization()
        self.bn2=BatchNormalization()
        self.relu2= ReLU()

    # use training param to control tensor flow type, e.g. BN or dropout layers are not called in testing
    def call(self, x, training=True):
        h=self.relu1(self.bn1(self.conv1(x),training))
        h=self.bn2(self.conv2(h),training)
        if self.conv3:
            x= self.conv3(x)
        y = self.relu2(h+x)
        return y

    def get_config(self):
        config = super(InceptionBlock, self).get_config()
        # config.update({"units": self.units})
        return config

# define a model
class GoogleNet(Model):
    def __init__(self, num_of_classes=10,**kwargs):
        super(GoogleNet, self).__init__(**kwargs)
        self.num_of_classes = num_of_classes

    # input_shape is (B,D1,...)
    def build(self, input_shape):
        self.conv1 = Conv2D(64, input_shape=input_shape,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.pool1 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same'
                                  )
        
        self.block1= self.resnet_block(64, 2, first_block=True)
        self.block2= self.resnet_block(128, 2)
        self.block3= self.resnet_block(256, 2)
        self.block4= self.resnet_block(512, 2)
        self.avg1 = GlobalAveragePooling2D()
        self.fc1 = Dense(self.num_of_classes, activation='softmax')

    def call(self, x, training=True):
        h=self.relu1(self.bn1(self.conv1(x),training))
        h=self.pool1(h)
        h=self.block1(h)
        h=self.block2(h)
        h=self.block3(h)
        h=self.block4(h)
        h=self.avg1(h)
        y=self.fc1(h)
        return y

    def resnet_block(self,num_channels, num_residuals, first_block=False):
        blk = keras.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(ResidualBlock(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(ResidualBlock(num_channels))
        return blk

    # overcome the bug show `multiple for tensor shape` for model.summary() in tf 2.1/2; 
    # tf2.3 fixed this bug already
    # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
    def summary2(self, x):
        x = Input(shape=x.shape[1:])
        self.build(x.shape)
        y = self.call(x)
        model = Model(inputs=[x], outputs=[y])
        return model.summary()

if __name__ == "__main__":

    # test ResidualBlock
    # blk = ResidualBlock(3)
    # blk = ResidualBlock(6, use_1x1conv=True, strides=2) # the last parameters should show together
    # Y = blk(X)
    # print(Y)

    x = np.random.uniform(size=(1, 224, 224, 3))
    model = GoogleNet()
    y = model(x)
    print(y)
    print(model.summary())
    # use my own summary method
    print(model.summary2(x))