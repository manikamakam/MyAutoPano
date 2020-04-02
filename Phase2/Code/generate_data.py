#!/usr/bin/evn python

"""
Author(s): 
Akanksha Patel
M.Eng in Robotics,
University of Maryland, College Park

Sri Manika Makam
M.Eng in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import os
import numpy as np
from numpy import savez
import cv2
from random import randrange
import matplotlib.pyplot as plt
# Add any python libraries here



def main():
        if os.path.exists("TxtFiles/LabelsVal.txt"):
            os.remove("TxtFiles/LabelsVal.txt")
        f = open("TxtFiles/LabelsVal.txt", "a")
	for i in range(1000):
	
		img  = cv2.imread("../Data/ValRaw/" + str(i+1)+".jpg")
	 
		img = cv2.resize(img, (240,320))
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		x = randrange(42,gray.shape[1]-42-128)
		y = randrange(42,gray.shape[0]-42-128)
		corners_A = [[y,x]]
		corners_A.append([y+128,x])
		corners_A.append([y+128,x+128])
		corners_A.append([y,x+128])
		
		patch_A = gray[y:y+128,x:x+128]
		A_normal=(patch_A-np.mean(patch_A))/255

		corners_B = []
		for j in corners_A:
		        new_x = randrange(0,32)
			new_y = randrange(0,32)
			corners_B.append([j[0]+new_x, j[1]+new_y])

		t = cv2.getPerspectiveTransform(np.float32(corners_A), np.float32(corners_B))

		t_inv= np.linalg.inv(t)     

		warpped_img = cv2.warpPerspective(gray, t_inv, (gray.shape[1], gray.shape[0]))
		
		patch_B = warpped_img[corners_A[0][0]:corners_A[2][0], corners_A[0][1]:corners_A[2][1]]
		B_normal=(patch_B-np.mean(patch_B))/255

		stacked_img = np.dstack((A_normal, B_normal))
		savez("../Data/Val/"+str(i+1)+".npz", stacked_img)


		label=[]
		for k in range(len(corners_A)): 
			label.append([corners_B[k][0]-corners_A[k][0], corners_B[k][1]-corners_A[k][1]])

		label = np.reshape(label,8)
		# print(label)
		# if os.path.exists("TxtFiles/LabelsTrain.txt"):
            #     os.remove("TxtFiles/LabelsTrain.txt")
            # f = open("TxtFiles/LabelsTrain.txt", "a")
            # f = open("TxtFiles/LabelsVal.txt", "a")
		np.savetxt(f, label , fmt="%s", newline=' ')
		f.write("\n")
	        



    
if __name__ == '__main__':
    main()
 
