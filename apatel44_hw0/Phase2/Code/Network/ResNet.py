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

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models
# Don't generate pyc codes
sys.dont_write_bytecode = True

def conv_block(net, filters, flag):
    shortcut = net

    # print(net.shape)

    # Layer 1
    if (not flag):
        net = tf.layers.conv2d(inputs = net, padding='same',filters = filters, kernel_size = 3, activation = None)
    else:
        net = tf.layers.conv2d(inputs = net, padding='same',filters = filters, kernel_size = 3, activation = None, strides = 2)

    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True)
    net = tf.nn.relu(net)
    # net = tf.nn.dropout(net, rate = 0.5)
    
    # Layer 2
    net = tf.layers.conv2d(inputs = net, padding='same',filters = filters, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True)
    net = tf.nn.relu(net)
    # net = tf.nn.dropout(net, rate = 0.5)

    if(not flag):
        # print(net.shape)

        net = tf.keras.layers.add([net, shortcut])
        net = tf.nn.relu(net)

    shortcut = net

    # Layer 3
    net = tf.layers.conv2d(inputs = net, padding='same',filters = filters, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True,)
    net = tf.nn.relu(net)
    # net = tf.nn.dropout(net, rate = 0.5)

    # Layer 4
    net = tf.layers.conv2d(inputs = net, padding='same',filters = filters, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True,)
    net = tf.nn.relu(net)
    # net = tf.nn.dropout(net, rate = 0.5)    

    # print(net.shape)

    net = tf.keras.layers.add([net, shortcut])
    net = tf.nn.relu(net)

    return net


def resnet(x, num_classes, input_shape):
    img_input = layers.Input(shape=input_shape)
 
    net = tf.layers.conv2d(inputs = x, padding='same', strides=(2, 2), filters = 64, kernel_size = 7, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True,)
    net = tf.nn.relu(net)
    # net = tf.nn.dropout(net, rate = 0.5)

    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2, name = 'max_pooling1')

    net = conv_block(net, 64, False)

    net = conv_block(net, 128, True)

    net = conv_block(net, 256, True)

    net = conv_block(net, 512, True)

    net = tf.keras.layers.GlobalAveragePooling2D()(net)

    net = tf.layers.flatten(net)
    # net = tf.nn.dropout(net, rate = 0.5)

    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = num_classes, activation = None)

    #prLogits is defined as the final output of the neural network
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax

