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
from random import randrange
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
    NumImages = len(glob.glob(BasePath+'/Data/P1TestSet/Phase2/*.jpg'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + '/Data/P1TestSet/Phase2/' + str(count) + '.jpg')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath, count):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """

    ModelType = 'Unsup'
    
    if (ModelType  == 'Sup'):
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
       
    else:
        ImageName = DataPath

	stacked_img = []
	PABatch  = []
	CABatch  = []
	PBBatch  = []

        img = cv2.imread(ImageName)
        img = cv2.resize(img, (240,320))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = (gray-np.mean(gray))/255
    

        x = randrange(42, gray.shape[1]-42-128)
        y = randrange(42, gray.shape[0]-42-128)
        corners_A = [[y,x]]
        corners_A.append([y+128, x])
        corners_A.append([y+128, x+128])
        corners_A.append([y, x+128])
        
        patch_A = gray[y:y+128, x:x+128]
        A_normal = (patch_A-np.mean(patch_A))/255
        
        corners_B = []
        for j in corners_A:
            new_x = randrange(0,32)
            new_y = randrange(0,32)
            corners_B.append([j[0]+new_x, j[1]+new_y])
    

        t = cv2.getPerspectiveTransform(np.float32(corners_A), np.float32(corners_B))
        t_inv = np.linalg.inv(t)  
        warpped_img = cv2.warpPerspective(gray, t_inv, (gray.shape[1], gray.shape[0]))
        
        patch_B = warpped_img[corners_A[0][0]:corners_A[2][0], corners_A[0][1]:corners_A[2][1]]
        B_normal=(patch_B-np.mean(patch_B))/255

        stacked_img.append(np.dstack((A_normal, B_normal)))

        PABatch.append(np.float32(A_normal.reshape(128,128,1)))
        CABatch.append(np.float32(corners_A))
        PBBatch.append(np.float32(B_normal.reshape(128,128,1)))

        return stacked_img, PABatch, CABatch, PBBatch
            

def TestOperation(ImgPH, PatchAPH, CornerAPH, PatchBPH, ImageSize, ModelPath, DataPath, Epochs):
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
    # Predict output with forward pass, MiniBatchSize for Test is 1
    prLogits , _, _, _, _, _, _, _, _, _  = UnsupervisedModel(ImgPH, PatchAPH, CornerAPH, PatchBPH, ImageSize, 1)
    loss = 255 * tf.reduce_mean(tf.abs(prLogits - PatchBPH))

    loss_summary = tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('PatchBPH', PatchBPH)
    tf.summary.image('PredB', prLogits)
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()
    LogsPath = "/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2/Code/Logs/Unsup/Test/"

    appendLoss = []
    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            stacked_img, PABatch, CABatch, PBBatch = ReadImages(ImageSize, DataPathNow, count)
            FeedDict = {ImgPH: stacked_img, PatchAPH: PABatch, CornerAPH: CABatch, PatchBPH: PBBatch}

            pred_PB, loss_fetched, Summary = sess.run([prLogits, loss, MergedSummaryOP], FeedDict)

            appendLoss.append(loss_fetched)

            Writer.add_summary(Summary, Epochs*1000 + count)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

            cv2.imwrite("../Data/Test/Unsup/Pred/"+str(count+1)+".jpg", pred_PB)

        return np.mean(appendLoss)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = []
        LabelPath = "../Data/Test/Unsup/Label/"
        for name in sorted(os.listdir(LabelPath)):
            im = cv2.imread(os.path.join(LabelPath, name));
            if im is not None:
                LabelTest.append(im)

        LabelTest = np.asarray(LabelTest)

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = []
        PredPath = "../Data/Test/Unsup/Pred/"
        for name in sorted(os.listdir(PredPath)):
            im = cv2.imread(os.path.join(PredPath, name));
            if im is not None:
                LabelPred.append(im)

        LabelPred = np.asarray(LabelPred)
        
    return LabelTest, LabelPred

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2/Checkpoints/Unsup/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    NumEpochs = 50
    appendloss = []
    for epoch in range(NumEpochs):
        # Parse Command Line arguments
        tf.reset_default_graph()

        ModelPath = '/home/p_akanksha94/CMSC733/apatel44_smakam_p1/Phase2/Checkpoints/Unsup/'+str(epoch)+'model.ckpt'

        # Setup all needed parameters including file reading
        ImageSize, DataPath = SetupAll(BasePath)

        # Define PlaceHolder variables for Input and Predicted output
        ImgPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
        LabelPH = tf.placeholder(tf.float32, shape=(1, 8)) # OneHOT labels
        PatchAPH = tf.placeholder(tf.float32, shape=(1, 128, 128, 1))
        CornerAPH = tf.placeholder(tf.float32, shape=(1, 4, 2))
        PatchBPH = tf.placeholder(tf.float32, shape=(1, 128, 128, 1))

        loss = TestOperation(ImgPH, PatchAPH, CornerAPH, PatchBPH, ImageSize, ModelPath, DataPath, epoch)

        print("Loss:", loss)
        
if __name__ == '__main__':
    main()
 
