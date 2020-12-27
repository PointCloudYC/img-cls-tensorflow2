"""
LeNet implementation in TF2 
Date: Dec 2, 20
Author: YIN Chao
"""

import time
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import \
Input, Dense, Conv2D, Conv1D, MaxPooling2D, \
Add, Activation, Flatten, \
BatchNormalization, AveragePooling2D, GlobalAveragePooling2D

import numpy as np


# define a model
class LeNet(Model):
    def __init__(self, filters=[6, 16], kernel_size=5, activation='sigmoid', num_of_class=10, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.num_of_class = num_of_class

    # input_shape is (B,D1,...)
    def build(self, input_shape):
        self.conv1 = Conv2D(
            self.filters[0],
            kernel_size=self.kernel_size,
            padding='same',
            activation=self.activation,
        )
        self.pooling1 = AveragePooling2D(pool_size=(2, 2), strides=2)

        self.conv2 = Conv2D(
            self.filters[1],
            kernel_size=self.kernel_size,
            padding='valid',
            activation=self.activation
        )
        self.pooling2 = AveragePooling2D(pool_size=(2, 2), strides=2)

        self.flat = Flatten()

        self.fc1 = Dense(120, activation=self.activation)
        # self.dp1 = keras.layers.Dropout(0.5)
        self.fc2 = Dense(84, activation=self.activation)
        # self.dp2 = keras.layers.Dropout(0.5)
        self.fc3 = Dense(self.num_of_class, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.flat(x)
        x = self.fc1(x)
        # x= self.dp1(x)
        x = self.fc2(x)
        # x= self.dp2(x)
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
    x = np.random.randn(2, 28, 28, 1)
    model = LeNet(filters=[6, 16], kernel_size=5, activation='sigmoid')
    y = model(x)
    print(y)
    print(model.summary())
    # use my own summary method
    print(model.summary2(x))