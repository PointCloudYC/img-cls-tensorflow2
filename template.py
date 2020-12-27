import os
import time
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import \
    Input, GlobalAveragePooling2D, Add, \
    Dense, Activation, Flatten, \
    BatchNormalization, Conv2D, MaxPooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print(tf.__version__)

np.random.seed(123)
tf.random.set_seed(123)


"""
define a custom layer
- a layer essentially encapsulate a tensor computation
- 3 methods: __init__() to define attributes(e.g. name,w,b;for better maintenance, w,b often are intialized  in build method), build() to initialize tensors, and call() methods(return the output tensor, pay special attention to the shape flow)
- layer can has non-trainable variables by decalring the trainable property
- often, it is highly suggested to defer layer input
- add_loss/add_metric() to add a tensor to layer.losses or metrics collection for future use;
- enable to serialize layer states using get_config() 
check: https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""

class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    # use training param to control tensor flow type, e.g. BN or dropout layers are not called in testing
    def call(self, inputs, training=True):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config

x=np.random.randn(2,9)
# call __init__ method
layer = Linear(units=32)
# when layer() it will trigger build method to build tensor dynamically, then trigger call method
y=layer(x)
print(y)

# serialization
layer = Linear(64)
config = layer.get_config()
print(config)
# new_layer = Linear.from_config(config)




"""
define custom training loop 
refs:
* [Writing a training loop from scratch  |  TensorFlow Core](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
* [Custom training: walkthrough  |  TensorFlow Core](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)
"""
