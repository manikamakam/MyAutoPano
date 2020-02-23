#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
# import matplotlib.pyplot as plt
from Network.Network import SupervisedModel, UnsupervisedModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
from random import randrange
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, ModelType):
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    if (ModelType  == 'Sup'):
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        I1Batch = []
        LabelBatch = []

        
        ImageNum = 0
        while ImageNum < MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(DirNamesTrain)-1)
            
            
            RandImageName = BasePath + os.sep +'Data' + os.sep + DirNamesTrain[RandIdx] + '.npz'   
            ImageNum += 1
            
            ##########################################################
            # Add any standardization or data augmentation here!
            ##########################################################

            # I1 = np.float32(cv2.imread(RandImageName))
            I1 = np.load(RandImageName, allow_pickle = False)["arr_0"]
            # Label = convertToOneHot(TrainLabels[RandIdx], 10)
            Label = [0] * 8
            Label[0]= TrainLabels[RandIdx*8]
            for i in range(7):
                Label[i+1] = TrainLabels[RandIdx*8 + i + 1]

            # print(Label.shape)
            # print("RandIdx: ", RandIdx)
            # print(Label)
            # Append All Images and Mask
            I1Batch.append(I1)
            LabelBatch.append(Label)
            
        return I1Batch, LabelBatch
    else:
        print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        StackedBatch = []
        IABatch =[]
        CABatch =[]
        PBBatch =[]


        ImageNum = 0
        while ImageNum < MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(DirNamesTrain)-1)
            
            
            RandImageName = BasePath + os.sep +'Data' + os.sep + 'Raw_'+ DirNamesTrain[RandIdx] + '.jpg'   
            # print(RandImageName)
            ImageNum += 1

            img = cv2.imread(RandImageName)
            img = cv2.resize(img, (240,320))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray=(gray-np.mean(gray))/(np.std(gray) + 10**(-7))
        

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
        

            t = cv2.getPerspectiveTransform(np.float32(corners_B), np.float32(corners_A))
            warpped_img = cv2.warpPerspective(gray, t, (gray.shape[1], gray.shape[0]))
            
            patch_B = warpped_img[corners_A[0][0]:corners_A[2][0], corners_A[0][1]:corners_A[2][1]]
            B_normal=(patch_B-np.mean(patch_B))/(np.std(patch_B) + 10**(-7))

            stacked_img = np.dstack((A_normal, B_normal))

            StackedBatch.append(stacked_img)
            IABatch.append(np.float32(gray.reshape(320,240,1)))
            CABatch.append(np.float32(corners_A))
            PBBatch.append(np.float32(B_normal.reshape(128,128,1)))

        return StackedBatch, IABatch, CABatch, PBBatch

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
def TrainOperation(ImgPH, LabelPH,IAfullPH,CornerAPH, PatchBPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Predict output with forward pass


    if ModelType == 'Sup':
        print("Supervised")
        prLogits = SupervisedModel(ImgPH, ImageSize, MiniBatchSize)

        with tf.name_scope('Loss'):
            loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(prLogits,LabelPH))))

        with tf.name_scope('Adam'):
            Optimizer = tf.train.AdamOptimizer(learning_rate=0.0005,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(loss)

    else:
        print("Unsupervised")
        pred_PB,PB = UnsupervisedModel(ImgPH,IAfullPH,CornerAPH, PatchBPH,ImageSize, MiniBatchSize)

        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.abs(pred_PB - PB))


        with tf.name_scope('Adam'):
            Optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    EpochLossPH = tf.placeholder(tf.float32, shape=None)
    loss_summary = tf.summary.scalar('LossEveryIter', loss)
    epoch_loss_summary = tf.summary.scalar('LossPerEpoch', EpochLossPH)
    # tf.summary.image('Anything you want', AnyImg)

    # Merge all summaries into a single operation
    MergedSummaryOP1 = tf.summary.merge([loss_summary])
    MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])
    # MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver(max_to_keep=10000)
    AccOverEpochs=np.array([0,0])
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            Loss=[]
            epoch_loss=0
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                if ModelType == "Sup":
                    ImgBatch,LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType)
                    FeedDict = {ImgPH: ImgBatch, LabelPH: LabelBatch}
                    
                else:
                    StackedBatch, IABatch, CABatch, PBBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType)
                    FeedDict = {ImgPH: StackedBatch, IAfullPH:IABatch ,CornerAPH: CABatch, PatchBPH: PBBatch}
                    
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
                Loss.append(LossThisBatch)
                epoch_loss = epoch_loss + LossThisBatch
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
              

            epoch_loss = epoch_loss/NumIterationsPerEpoch
            
            print(np.mean(Loss))
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
            Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
            Writer.add_summary(Summary_epoch,Epochs)
            Writer.flush()
            

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/akanksha/Downloads/Phase2', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/Unsup/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/Unsup/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType
    print(ModelType)

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)



    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
    IAfullPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 320,240,1))
    CornerAPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 4,2))
    PatchBPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128,1))
    
    
    TrainOperation(ImgPH, LabelPH,IAfullPH,CornerAPH, PatchBPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
        
    
if __name__ == '__main__':
    main()
 
