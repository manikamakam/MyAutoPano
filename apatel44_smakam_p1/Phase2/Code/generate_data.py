#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 2 Data Generation Code

Author(s): 
Akanksha Patel
M.Eng in Robotics,
University of Maryland, College Park

Sri Manika Makam
M.Eng in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
from numpy import savez
import cv2
from random import randrange
import matplotlib.pyplot as plt
# Add any python libraries here



def main():
	for i in range(5000):
	
		img  = cv2.imread("/home/manika/Desktop/CMSC733/smakam_p1/Phase2/Data/Raw_Train/" + str(i+1)+".jpg")
	 
		# resize image
		img = cv2.resize(img, (240,320))
		# print(img.shape)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# img2  = cv2.imread("/home/manika/Desktop/CMSC733/smakam_p1/Phase2/Data/Train/2.jpg")
		# print(gray.shape)
		# cv2.imshow('gray', gray)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		x = randrange(42,gray.shape[1]-42-128)
		y = randrange(42,gray.shape[0]-42-128)
		corners_A = [[y,x]]
		corners_A.append([y+128,x])
		corners_A.append([y+128,x+128])
		corners_A.append([y,x+128])
		
		patch_A = gray[y:y+128,x:x+128]
		A_normal=(patch_A-np.mean(patch_A))/(np.std(patch_A) + 10**(-7))
		# print(np.mean(A_normal))
		# print(np.std(A_normal))
		# print(patch_A.shape)
		# print(corners_A)
		# cv2.imshow('Patch A',patch_A )
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		corners_B = []
		for j in corners_A:
			new_x = randrange(0,32)
			new_y = randrange(0,32)
			corners_B.append([j[0]+new_x, j[1]+new_y])
		# print(corners_B)

		t = cv2.getPerspectiveTransform(np.float32(corners_B), np.float32(corners_A))
		# print(t)

		# t_inv= np.linalg.inv(t)
		# print(t_inv)

		warpped_img = cv2.warpPerspective(gray, t, (gray.shape[1], gray.shape[0]))
		
		patch_B = warpped_img[corners_A[0][0]:corners_A[2][0], corners_A[0][1]:corners_A[2][1]]
		B_normal=(patch_B-np.mean(patch_B))/(np.std(patch_B) + 10**(-7))
		# print(np.mean(B_normal))
		# print(np.std(B_normal))
		# cv2.imshow('Patch B', patch_B)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		stacked_img = np.dstack((A_normal, B_normal))
		# print(stacked_img.shape) 
		savez("/home/manika/Desktop/CMSC733/smakam_p1/Phase2/Data/Train/"+str(i+1)+".npz", stacked_img)


		label=[]
		for k in range(len(corners_A)): 
			label.append([corners_A[k][0]-corners_B[k][0], corners_A[k][1]-corners_B[k][1]])

		label = np.reshape(label,8)
		# print(label)
		f = open("/home/manika/Desktop/CMSC733/smakam_p1/Phase2/Code/TxtFiles/LabelsTrain.txt", "a")
		np.savetxt(f, label , fmt="%s", newline=' ')
		f.write("\n")






	



    
if __name__ == '__main__':
    main()
 
