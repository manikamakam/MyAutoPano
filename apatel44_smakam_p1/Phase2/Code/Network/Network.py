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
from Misc.TFSpatialTransformer import transformer
from Misc.TensorDLT import TensorDLT
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SupervisedModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    train = True

    #############################
    # Fill your network here!
    #############################
    x = tf.layers.conv2d(inputs=Img, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1, name='conv1', \
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm1')
    x = tf.nn.relu(x, name='relu1')

    kernel_c1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/bias')[0]


    # kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
    # bias = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/bias')[0]
    # kernel=tf.transpose(kernel,[3,0,1,2])
    # kernel = kernel[:,:,:,1]
    # kernel = tf.reshape(kernel, [64, 3, 3, 1]) 

    # print(kernel)
    # print(bias)


    x = tf.layers.conv2d(inputs=x, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1, name='conv2',\
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm2')
    x = tf.nn.relu(x, name='relu2')
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='VALID', name='mx_pooling2')

    kernel_c2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')
    #bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]

    x = tf.layers.conv2d(inputs=x, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1, name='conv3',\
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm3')
    x = tf.nn.relu(x, name='relu3')
    
    kernel_c3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]


    x = tf.layers.conv2d(inputs=x, padding='same', filters=64, kernel_size=[3,3], activation=None, strides = 1, name='conv4',\
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm4')
    x = tf.nn.relu(x, name='relu4')
    x=tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='VALID', name='mx_pooling4')

    kernel_c4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv41/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1, name='conv5',\
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm5')
    x = tf.nn.relu(x, name='relu5')

    # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1, name='conv6',\
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm6')
    x = tf.nn.relu(x, name='relu6')
    x=tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='VALID', name='mx_pooling6')
    
    # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1, name='conv7',\
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm7')
    x = tf.nn.relu(x, name='relu7')

    # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]


    x = tf.layers.conv2d(inputs=x, padding='same', filters=128, kernel_size=[3,3], activation=None, strides = 1, name='conv8',\
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm8')
    x = tf.nn.relu(x, name='relu8')
    x = tf.nn.dropout(x, keep_prob=0.5, name='dropout1')
 
    # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]
   
    x = tf.layers.flatten(x, name='flatten')  

    x = tf.layers.dense(inputs=x, name='fc_1',units=1024, activation=tf.nn.relu, \
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.layers.batch_normalization(x, training=train, name='batch_norm9')
    # x = tf.nn.relu(x, name='relu9')
    x = tf.nn.dropout(x, keep_prob=0.5, name='dropout2')

    kernel_f1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/kernel')
    bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]

    x = tf.layers.dense(inputs=x, name='fc_2',units=8, activation= None)

    kernel_f2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_2/kernel')
    # bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc_1/bias')[0]


    #prLogits is defined as the final output of the neural network
    prLogits = x
    
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    # prSoftMax = tf.nn.softmax(logits = prLogits)
    
    return prLogits, kernel_c1, kernel_c2, kernel_c3, kernel_c4, kernel_f1, kernel_f2, bias

def UnsupervisedModel(Img,IA,CA, PB,ImageSize, MiniBatchSize):
    H4pt, kernel_c1, kernel_c2, kernel_c3, kernel_c4, kernel_f1, kernel_f2, bias = SupervisedModel(Img, ImageSize, MiniBatchSize)
    C4A_pts = tf.reshape(CA,[MiniBatchSize,8])
   
    H_mat = TensorDLT(H4pt, C4A_pts, MiniBatchSize)
    img_h = ImageSize[0]
    img_w = ImageSize[1]
    # Constants and variables used for spatial transformer
    M = np.array([[img_w/2.0, 0., img_w/2.0],
              [0., img_h/2.0, img_h/2.0],
              [0., 0., 1.]]).astype(np.float32)

    M_tensor  = tf.constant(M, tf.float32)
    M_tile   = tf.tile(tf.expand_dims(M_tensor, [0]), [MiniBatchSize, 1,1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv   = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [MiniBatchSize,1,1])

    y_t = tf.range(0, MiniBatchSize*img_w*img_h, img_w*img_h)
    z =  tf.tile(tf.expand_dims(y_t,[1]),[1,128*128])
    batch_indices_tensor = tf.reshape(z, [-1]) # Add these value to patch_indices_batch[i] for i in range(num_pairs) # [BATCH_SIZE*WIDTH*HEIGHT]

    # Transform H_mat since we scale image indices in transformer
    H_mat = tf.matmul(tf.matmul(M_tile_inv, H_mat), M_tile)
    # Transform image 1 (large image) to image 2
    out_size = (img_h, img_w)

    
    warped_images, _ = transformer(IA, H_mat, out_size)
    # print(warped_images.get_shape())
    pred_PB = tf.reduce_mean(warped_images, 3)

    pred_PB = tf.reshape(pred_PB, [MiniBatchSize, 128, 128, 1])


    return pred_PB, PB, kernel_c1, kernel_c2, kernel_c3, kernel_c4, kernel_f1, kernel_f2, bias
