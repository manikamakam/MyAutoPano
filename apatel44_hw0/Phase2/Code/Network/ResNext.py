#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Akanksha Patel (apatel44@umd.edu)
Masters of Engineering in Robotics,
University of Maryland, College Park
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models
# Don't generate pyc codes
sys.dont_write_bytecode = True

# def bottleneck_layer(x, scope, filters):
#     with tf.name_scope(scope):
#         print("Inside BottleNeck")
#         print(x.shape)
#         x = tf.layers.batch_normalization(inputs = x,axis = -1, center = True, scale = True, training = True)
#         x = tf.nn.relu(x)
#         x = tf.nn.dropout(x, rate = 0.5)
#         x = tf.layers.conv2d(inputs = x, padding='same',filters = 4 * filters, kernel_size = 1, activation = None)

#         print(x.shape)   
#         x = tf.layers.batch_normalization(inputs = x,axis = -1, center = True, scale = True, training = True)
#         x = tf.nn.relu(x)
#         x = tf.nn.dropout(x, rate = 0.5)
#         x = tf.layers.conv2d(inputs = x, padding='same',filters = filters, kernel_size = 3, activation = None)
#         print(x.shape)
        
#         return x

# def dense_block(input_x, nb_layers, layer_name, filters):
#     with tf.name_scope(layer_name):
#         layers_concat = list()
#         layers_concat.append(input_x)

#         print("Inside Dense")
#         print(input_x.shape)
#         x = bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0), filters = filters)

#         layers_concat.append(x)

#         for i in range(nb_layers - 1):
#             x = tf.concat(layers_concat, axis=3)
#             x = bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1), filters = filters)
#             layers_concat.append(x)

#         x = tf.concat(layers_concat, axis=3)        
#         return x

# def transition_layer(x, scope, filters):
#     print("Inside Transition")
#     with tf.name_scope(scope):
#         x = tf.layers.batch_normalization(inputs = x,axis = -1, center = True, scale = True, training = True)
#         x = tf.nn.relu(x)
#         # x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
#         # x = Relu(x)
#         # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
        
#         # https://github.com/taki0112/Densenet-Tensorflow/issues/10
#         print(x.shape)
#         in_channel = x.shape[-1]
#         x = tf.layers.conv2d(inputs = x, padding='same',filters = 2*filters, kernel_size = 7, activation = None)
#         print(x.shape)
#         x = tf.nn.dropout(x, rate = 0.5)
#         x = tf.keras.layers.AveragePooling2D()(x)
#         print("After 2D pooling")
#         print(x.shape)
#         # x = conv_layer(x, filter=in_channel*0.5, kernel=[1,1], layer_name=scope+'_conv1')
#         # x = Drop_out(x, rate=dropout_rate, training=self.training)
#         # x = Average_pooling(x, pool_size=[2,2], stride=2)

        # return x

def resnext_block(im, filters , layer_name):
    net = tf.layers.conv2d(inputs = im, padding='same',filters = filters, kernel_size = 1, activation = None)
    net = tf.layers.batch_normalization(inputs = net, axis = -1, center = True, scale = True)
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, 0.5)

    net = tf.layers.conv2d(inputs = net, padding='same',filters = filters, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net, axis = -1, center = True, scale = True)
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, 0.5)

    net = tf.layers.conv2d(inputs = net, padding='same',filters = 2*filters, kernel_size = 1, activation = None)
    net = tf.layers.batch_normalization(inputs = net, axis = -1, center = True, scale = True)
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, 0.5)

    # net = tf.keras.layers.add([net, net, net, im])
    return net


def resnext(img, num_classes, input_shape):
    # img_input = layers.Input(shape=input_shape)

 
    # Conv1 (7x7,64,stride=2)
    
    # net = tf.layers.ZeroPadding2D(padding=(3, 3))(net)
    # filters = 24

    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # print(x.shape)
    x = tf.layers.conv2d(inputs = img, padding='same',filters = 64, kernel_size = 7, activation = None)
    # print(x.shape)
    x = tf.layers.max_pooling2d(inputs = x, pool_size = 3, strides = 2, name = 'max_pooling1')
    # x = conv_layer(x, filter=2*filters, kernel=[7,7], stride=2, layer_name='conv0')
    # x = Max_Pooling(x, pool_size=[3,3], stride=2)

    # print(x.shape)
    t1 = resnext_block(im=x, filters = 32, layer_name='dense_1')
    t2 = resnext_block(im=x, filters = 32, layer_name='dense_1')
    t3 = resnext_block(im=x, filters = 32, layer_name='dense_1')


    x1 = tf.keras.layers.add([t1, t2])
    x2 = tf.keras.layers.add([t3, x])
    x = tf.keras.layers.add([x1, x2])

    t1 = resnext_block(im=x, filters = 64, layer_name='dense_1')
    t2 = resnext_block(im=x, filters = 64, layer_name='dense_1')
    t3 = resnext_block(im=x, filters = 64, layer_name='dense_1')
    t3 = resnext_block(im=x, filters = 64, layer_name='dense_1')

    x = tf.keras.layers.AveragePooling2D()(x)

    x = tf.layers.flatten(x)
    x = tf.nn.dropout(x, 0.5)

    x = tf.layers.dense(inputs = x, name='layer_fc_out', units = num_classes, activation = None)

    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    prLogits = x
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax

