#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 2 Starter Code

Author(s): 
Akanksha Patel
M.Eng in Robotics,
University of Maryland, College Park

Sri Manika Makam
M.Eng in Robotics,
University of Maryland, College Park
"""

import tensorflow as tf
import cv2
import os
import sys
import glob
import Misc.ImageUtils as iu
from random import randrange
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import UnsupervisedModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
from Misc.TensorDLT import *
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


# Don't generate pyc codes
sys.dont_write_bytecode = True



def generate_data(BasePath):
    StackedBatch = []
    PABatch =[]
    CABatch =[]
    PBBatch =[]
    RandIdx = random.randint(1, 1000)

    RandImageName = BasePath + str(RandIdx) + '.jpg'   
    img = cv2.imread(RandImageName)
    img = cv2.resize(img, (240,320))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    x = randrange(42,gray.shape[1]-42-128)
    y = randrange(42,gray.shape[0]-42-128)
    corners_A = [[y,x]]
    corners_A.append([y+128,x])
    corners_A.append([y+128,x+128])
    corners_A.append([y,x+128])
            
    patch_A = gray[y:y+128,x:x+128]
    A_normal=(patch_A-np.mean(patch_A))/(np.std(patch_A) + 10**(-7))
            
    corners_B = []
    for j in corners_A:
        new_x = randrange(0,32)
        new_y = randrange(0,32)
        corners_B.append([j[0]+new_x, j[1]+new_y])
        
    print("B: ", corners_B)

    H = cv2.getPerspectiveTransform(np.float32(corners_A), np.float32(corners_B))
    H_inv = np.linalg.inv(H)
    warpped_img = cv2.warpPerspective(gray, H_inv, (gray.shape[1], gray.shape[0]))
            
    patch_B = warpped_img[corners_A[0][0]:corners_A[2][0], corners_A[0][1]:corners_A[2][1]]
    B_normal=(patch_B-np.mean(patch_B))/(np.std(patch_B) + 10**(-7))

    stacked = np.dstack((A_normal, B_normal))

    StackedBatch.append(stacked)
    PABatch.append(np.float32(A_normal.reshape(128,128,1)))
    CABatch.append(np.float32(corners_A))
    PBBatch.append(np.float32(B_normal.reshape(128,128,1)))

    return StackedBatch, PABatch, CABatch, PBBatch, corners_B,img


def TestOperation(ImgPH,PatchAPH,CornerAPH, PatchBPH, ImageSize, ModelPath, BasePath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    StackedBatch, PABatch, CABatch, PBBatch,corners_B,img = generate_data(BasePath)

    prLogits, PB, H4pt, _, _, _, _, _, _, _  = UnsupervisedModel(ImgPH,PatchAPH,CornerAPH, PatchBPH,ImageSize,1)

    Saver = tf.train.Saver()
  
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        FeedDict = {ImgPH: StackedBatch, PatchAPH:PABatch ,CornerAPH: CABatch, PatchBPH: PBBatch}
        Pred_H = sess.run(H4pt,FeedDict)

    print("old:" , Pred_H.shape)
    Pred_H = np.reshape(Pred_H, (4,2))
    print("new:", Pred_H.shape)
    cornersB_new = Pred_H + CABatch[0]

    print("cornersB_new:", cornersB_new.shape)

    
    for i in corners_B: 
        temp = i[0]
        i[0] = i[1]
        i[1] = temp
    print(corners_B) 

    for i in cornersB_new: 
        temp = i[0]
        i[0] = i[1]
        i[1] = temp

    cv2.polylines(img,np.int32([cornersB_new]),True,(0,0,255), 3)
    cv2.polylines(img,np.int32([corners_B]),True,(255,0,0), 3)

    cv2.imwrite('../Graphs/unsup/Val/4.jpg',img)



def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    # Parse Command Line arguments

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../archive/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../Data/Raw_Val/', help='Path to load images from, Default:BasePath')
    
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath

    ImageSize = [1,128,128,2]

    ImgPH = tf.placeholder(tf.float32, shape=(1, ImageSize[1], ImageSize[2], ImageSize[3]))
    PatchAPH = tf.placeholder(tf.float32, shape=(1, 128,128,1))
    CornerAPH = tf.placeholder(tf.float32, shape=(1, 4,2))
    PatchBPH = tf.placeholder(tf.float32, shape=(1, 128, 128,1))

    TestOperation(ImgPH,PatchAPH,CornerAPH, PatchBPH, ImageSize, ModelPath, BasePath)

    
if __name__ == '__main__':
    main()
 
