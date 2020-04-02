#!/usr/bin/env python

"""
Author(s):
Sri Manika Makam
M.Eng in Robotics,
University of Maryland, College Park

Akanksha Patel
M.Eng in Robotics,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
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


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [128, 128, 2]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'/Data/Val/*.npz'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + '/Data/Val/' + str(count) + '.npz')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = np.load(ImageName, allow_pickle = False)["arr_0"]
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    I1Combined = np.expand_dims(I1, axis=0)

    return I1Combined, I1
                

def TestOperation(ImgPH, LabelsTrue, ImageSize, ModelPath, DataPath, LabelsPath, LabelsPathPred, Epochs):
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
    Length = ImageSize[0]
    LabelsTrueAll = ReadLabels(LabelsPath)
    print(np.asarray(LabelsTrue).shape)
    print(LabelsTrue[0,:])
    # Predict output with forward pass, MiniBatchSize for Test is 1
    prLogits = HomographyModel(ImgPH, ImageSize, 1)[0]

    loss = tf.sqrt(tf.reduce_mean((tf.squared_difference(prLogits, LabelsTrue))))

    tf.summary.scalar('LossEveryIter', loss)
    # tf.summary.image('PatchBPH', PatchBPH)
    # tf.summary.image('PredB', prLogits)
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()
    LogsPath = "/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2-Sup/Code/Logs/Val"

    appendLoss = []

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        

        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            label = np.asarray(LabelsTrueAll[count])
            label = np.reshape(label, (1,8))
            Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img, LabelsTrue: label}
            PredT, loss_fetched, Summary = sess.run([prLogits, loss, MergedSummaryOP], FeedDict)

            appendLoss.append(loss_fetched)

            Writer.add_summary(Summary, Epochs*1000 + count)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        return np.mean(appendLoss)


def ReadLabels(LabelsPathTest):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = np.asarray(map(float, LabelTest.split()))
        LabelTest = np.reshape(LabelTest, (1000, 8))
        
    return LabelTest

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2-Sup/Checkpoints/Sup/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2-Sup/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsVal.txt', help='Path of labels file, Default:./TxtFiles/LabelsVal.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    NumEpochs = 50
    appendLoss = []
    for epoch in range(NumEpochs):
        # Parse Command Line arguments
        tf.reset_default_graph()
        
        ModelPath = '/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2-Sup/Checkpoints/Sup/'+str(epoch)+'model.ckpt'
        # ModelPath = '/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2/Checkpoints/Sup/49model.ckpt'

        # Setup all needed parameters including file reading
        ImageSize, DataPath = SetupAll(BasePath)

        # Define PlaceHolder variables for Input and Predicted output
        ImgPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
        LabelsTrue = tf.placeholder(tf.float32, [1, 8])
        LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels
 
        loss = TestOperation(ImgPH, LabelsTrue, ImageSize, ModelPath, DataPath, LabelsPath, LabelsPathPred, epoch)
        print("Loss:",np.mean(loss))
     
if __name__ == '__main__':
    main()
 

