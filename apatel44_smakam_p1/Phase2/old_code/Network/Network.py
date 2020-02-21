"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize):
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
    x = tf.layers.conv2d(inputs=Img, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = tf.layers.conv2d(inputs=x, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x=tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='VALID')


    x = tf.layers.conv2d(inputs=x, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = tf.layers.conv2d(inputs=x, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x=tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='VALID')


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x=tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='VALID')


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=0.5)
    
    x = tf.layers.flatten(x)  

    x = tf.layers.dense(inputs=x, name='fc_1',units=1024, activation=tf.nn.relu)
    x = tf.nn.dropout(x, keep_prob=0.5)

    x = tf.layers.dense(inputs=x, name='fc_2',units=8, activation= None)



    #prLogits is defined as the final output of the neural network
    prLogits = x
    
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    # prSoftMax = tf.nn.softmax(logits = prLogits)
    
    return prLogits
