"""
train a model and evaluate 
Date: Dec 2, 20
Author: YIN Chao
"""

import argparse
import logging
import os
import time
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# models
from models.lenet import LeNet
from models.alexnet import AlexNet
from models.resnet18 import ResNet18

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print(tf.__version__)

np.random.seed(123)
tf.random.set_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


"""
get data
"""
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))

"""
vis. data
"""
PLOT_COUNT =25
idx = np.random.choice(len(train_images),PLOT_COUNT, replace=False)
plt.figure(figsize=(10,10))
for i in range(PLOT_COUNT):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()


"""
preprocess data
"""
batch_size =32
# expand 1 dim for convolution, (60000,28,28) --> (_,28,28,1)
train_images = np.expand_dims(train_images,axis=-1)
test_images = np.expand_dims(test_images,axis=-1)
train_images = train_images / 255.0
test_images = test_images / 255.0
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(batch_size).prefetch(1)# make sure you always have one batch ready to serve

val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size).prefetch(1)# make sure you always have one batch ready to serve

print(train_dataset)
print(val_dataset)



"""
build a model
"""
model = LeNet()


"""
define loss, optimizers, metrics, etc.
"""
# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


"""
custom training and evaluate loop
"""
epochs=20
for epoch in range(epochs):
    
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
            if model.losses:
                loss_value+=sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * 32))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

"""
infere on new data
"""
