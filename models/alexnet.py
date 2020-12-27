"""
AlexNet implementation in TF2 
Date: Dec 2, 20
Author: YIN Chao
"""

import os
import time
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Input, GlobalAveragePooling2D, Add, \
    Dense, Activation, Flatten, \
    BatchNormalization, Conv2D, MaxPooling2D,\
    AveragePooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print(tf.__version__)
np.random.seed(123)
tf.random.set_seed(123)


# define a model
class AlexNet(Model):
    def __init__(self, filters=[96, 256, 384], kernel_sizes=[11, 5, 3], activation='relu', num_of_class=10, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.num_of_class = num_of_class

    # input_shape is (B,D1,...)
    def build(self, input_shape):
        self.conv1 = Conv2D(
            self.filters[0],
            kernel_size=self.kernel_sizes[0],
            padding='same',
            activation=self.activation,
            strides=4,
        )
        self.pooling1 = MaxPooling2D(pool_size=(3, 3), strides=2)
        self.conv2 = Conv2D(
            self.filters[1],
            kernel_size=self.kernel_sizes[1],
            padding='same',
            activation=self.activation,
        )
        self.pooling2 = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.conv3 = Conv2D(
            self.filters[2],
            kernel_size=self.kernel_sizes[2],
            padding='same',
            activation=self.activation
        )

        self.conv4 = Conv2D(
            self.filters[2],
            kernel_size=self.kernel_sizes[2],
            padding='same',
            activation=self.activation
        )

        self.conv5 = Conv2D(
            self.filters[2],
            kernel_size=self.kernel_sizes[2],
            padding='same',
            activation=self.activation
        )

        self.pooling3 = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.flat = Flatten()

        self.fc1 = Dense(4096, activation=self.activation)
        self.dp1 = keras.layers.Dropout(0.5)
        self.fc2 = Dense(4096, activation=self.activation)
        self.dp2 = keras.layers.Dropout(0.5)
        self.fc3 = Dense(self.num_of_class, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pooling3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        y = self.fc3(x)
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


if __name__ == "__main__":
    x = np.random.randn(2, 224, 224, 1)
    model = AlexNet()
    y = model(x)
    print(y)
    print(model.summary())
    # use my own summary method
    print(model.summary2(x))