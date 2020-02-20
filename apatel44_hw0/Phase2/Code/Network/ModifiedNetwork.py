"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Akanksha Patel
Masters of Engineering in Robotics,
University of Maryland, College Park
"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10ModelMod(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    num_classes = 10
    tf.random.set_random_seed(0) 

    net = Img

    net = tf.layers.conv2d(inputs = net, name='layer1_conv', padding='same',filters = 32, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='layer1_bn')
    net = tf.nn.relu(net, name = 'layer1_relu')
    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2, name = 'max_pooling1')

    net = tf.layers.conv2d(inputs = net, name = 'layer2_conv', padding= 'same', filters = 64, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer2_bn')
    net = tf.nn.relu(net, name = 'layer2_Relu')
    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2, name = 'max_pooling2')

    net = tf.layers.flatten(net)
    # net = tf.nn.dropout(net, rate = 0.5)

    #Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 100, activation = tf.nn.relu)
    # net = tf.nn.dropout(net, rate = 0.5)

    net = tf.layers.dense(inputs = net, name='layer_fc2', units = num_classes, activation = None)

    #prLogits is defined as the final output of the neural network
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax

