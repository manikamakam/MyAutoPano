#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano

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
import cv2
import matplotlib.pyplot as plt
import copy
import random
from skimage.feature import peak_local_max
import sys
# Add any python libraries here

def getCorners(gray, img, num, img_num):
	"""
	Get corners from an image
	img - imput image
	num - number of features required
	"""

	print("Extracting corners...")

	corners = cv2.goodFeaturesToTrack(gray,num,0.01,5)

	count = 0
	for i in corners:
		x, y = i.ravel()
		cv2.circle(img, (x,y), 3, 255, -1)

		# if count%5 == 0:
		# 	cv2.imshow('dst',img)
		# 	if cv2.waitKey(0) & 0xff == 27:
		# 	    cv2.destroyAllWindows()

		count = count+1

	cv2.imwrite('Images/corners' + img_num + '.png', img)

	return corners

def anms(gray, img, corners, num, img_num):
	"""
	Get evenly distributed best features
	gray
	img
	corners - 
	num - Number of features required
	img_num - Image name
	"""

	print("Selecting " + str(num) + " best corners.")

	l, _, _ = corners.shape
	
	r = l * [sys.maxsize]
	rr = []

	for i in range(l-1, -1, -1):
		ED = r[i]
		for j in range(i):
			x_i = int(corners[i,:,0])
			y_i = int(corners[i,:,1])
			x_j = int(corners[j,:,0])
			y_j = int(corners[j,:,1])

			ED = (x_j - x_i)**2 + (y_j - y_i)**2;

			if ED < r[i]:
				r[i] = ED

		rr.append((r[i], x_i, y_i))

	rr.sort(key=lambda x: x[0], reverse=True)

	N_best = []

	for i in range(0,num):
		N_best.append((rr[i][1], rr[i][2]))

	for i in N_best:
		cv2.circle(img, i, 3, 255, -1)

	# cv2.imshow('dst',img)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	cv2.imwrite('Images/anms' + img_num + '.png', img)

	return N_best

def getDescriptors(gray, img, N_best, img_num):
	"""

	"""
	print("Getting descriptors...")

	descs = []
	patches = []

	p, q, r = img.shape

	gray = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)

	gray = np.pad(gray, (20,20), mode = 'edge')

	for i in N_best:
		x = i[0]
		y = i[1]

		patch = gray[y:y+40,x:x+40]

		# patch_reshaped = cv2.resize(patch_blur, (8,8), interpolation = cv2.INTER_CUBIC)

		patch_reshaped = cv2.resize(patch, (8,8), interpolation = cv2.INTER_AREA)
		patches.append(patch_reshaped)
		desc = np.reshape(patch_reshaped, (64,1))

		desc = (desc - np.mean(desc))/(np.std(desc)+10**-7)
		descs.append(desc)

	for k in range(len(patches)):
		plt.subplot(len(patches)/5,5,k+1)
		plt.axis('off')
		plt.imshow(patches[k],cmap='gray')

	plt.savefig('Images/FD' + img_num + '.png')
	plt.close()

	return descs

def SSD(desc1, desc2):
	"""
	Compute sum of square distance between two descriptors
	"""
	# s = sum((desc1-desc2)**2)
	
	s = 0
	for i in range(len(desc1)):
		s = s + (desc1[i] - desc2[i])**2

	return s

def featureMatching(descs1, descs2, th):

	print("Matching features... ")
	matched = []
	cor1 = []
	cor2 = []
	indexes = []

	for i in range(len(descs1)):
		d1 = descs1[i]
		match1 = float('inf')
		match2 = float('inf')
		for j, d2 in enumerate(descs2):
			s = SSD(d1, d2)
			if s < match1:
				match2 = match1
				match1 = s
				index = j
			elif s < match2:
				match2 = s
		
		ratio = match1/match2
		if ratio < th:
			indexes.append([i, index])

	return indexes

def getProjection(H, pts):

	p = np.append(pts, [1], axis = 0)
	p = np.matrix.transpose(p)

	proj = np.matmul(H,p)

	t = (proj[2] + 10**-7)
	proj = np.delete(proj, 2)
	proj = proj/t

	return proj


def RANSAC(img1, img2, matched, pts1, pts2, k, t):
	"""
	data - A set of observations.
    model - A model to explain observed data points.
    n - Minimum number of data points required to estimate model parameters.
    k - Maximum number of iterations allowed in the algorithm.
    t - Threshold value to determine data points that are fit well by model.
    d - Number of close data points required to assert that a model fits well to data.
	"""

	print("RANSAC...")

	count = 0
	bestFit = None
	maxInliers = 0
	l = len(matched)
	indx = range(l)
	maxMatIndx = []
	inliers1 = []
	inliers2 = []

	while count < k:
		in1 = []
		in2 = []

		rand_points = random.sample(indx, 4)
		p1 = np.array([[0]*2]*4)
		p2 = np.array([[0]*2]*4)
		for i in range(4):
			p1[i] = pts1[matched[rand_points[i]][0]]
			p2[i] = pts2[matched[rand_points[i]][1]]

		p1 = p1.astype('float32')
		p2 = p2.astype('float32')

		H = cv2.getPerspectiveTransform(p1, p2)
		matIndex = []

		for j in range(l):
			index1 = matched[j][0]
			index2 = matched[j][1]
			proj_p2 = getProjection(H, pts1[index1])
			# t1 = (proj_p2[2] + +10**-7)
			# np.delete(proj_p2, 2)
			# proj_p2 = proj_p2/t1
			err = SSD(pts2[index2], proj_p2)
			if err < t:
				matIndex.append([index1, index2])
				in1.append(pts1[index1])
				in2.append(pts2[index2])

		if len(matIndex) > maxInliers:
			maxInliers = len(matIndex)
			bestFit = H
			maxMatIndx = matIndex
			inliers1 = in1
			inliers2 = in2

		count = count + 1

	print(bestFit)

	return maxMatIndx, inliers1, inliers2


def plotMatches(img1, img2, keypoints1, keypoints2, matched, name):
	matchedImg = np.array([])
	matchvec = []
	for m in matched:
		matchvec.append(cv2.DMatch(m[0], m[1], 1))

	matchedImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matchvec, matchedImg)

	print("Save Image")

	cv2.imwrite('Images/' + name + '.png', matchedImg)

	# cv2.imshow('dst',matchedImg)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	return matchedImg

def computeHomography(in1, in2):
	l = len(in1)
	# print(l)

	A = np.array([[0] * 9] * 2*l)
	for i in range(l):
		x = in1[i][0]
		y = in1[i][1]
		xp = in2[i][0]
		yp = in2[i][1]
		p1 = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
		p2 = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]
		A[2*i] = p1
		A[2*i + 1] = p2

	u, s, vh = np.linalg.svd(A)

	# print(A.shape)

	# print(vh.shape)

	H_falt = vh[len(vh)-1, :]

	H = np.reshape(H_falt, (3, 3))

	print("H: ")
	print(H)

	return H

def getPadding(img_new, img, pt, pt_new):
	r, c, _ = img.shape
	r_new, c_new, _ = img_new.shape

	xl = int(round(pt[0] - pt_new[0]))
	yu = int(round(pt[1] - pt_new[1]))

	print("xl: " + str(xl))
	print("yu: " + str(yu))

	xr = c_new + xl - c
	yb = r_new + yu - r

	print("xr: " + str(xr))
	print("yb: " + str(yb))

	return xl, xr, yu, yb




def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""

	img1  = cv2.imread("/home/akanksha/Documents/CMSC733/apatel44_smakam_p1/Phase1/Code/Images/output.jpg")
	# img1  = cv2.imread("/home/akanksha/Documents/CMSC733/apatel44_smakam_p1/Phase1/Data/Train/Set2/1.jpg")
	# img2  = cv2.imread("/home/akanksha/Documents/CMSC733/apatel44_smakam_p1/Phase1/Data/Train/Set2/2.jpg")
	img2  = cv2.imread("/home/akanksha/Documents/CMSC733/apatel44_smakam_p1/Phase1/Data/Train/Set2/3.jpg")

	# print(cv2.GetSize(img1))

	gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
	# gray3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

	print(gray1.dtype)

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	# dst = cv2.cornerHarris(gray1, 2, 3, 0.04)

	img1_copy = copy.deepcopy(img1)
	img2_copy = copy.deepcopy(img2)
	# img3_copy = copy.deepcopy(img3)

	corners1 = getCorners(gray1, img1_copy, 100, str(1))
	corners2 = getCorners(gray2, img2_copy, 100, str(2))
	# corners3 = getCorners(gray3, img3_copy, 3, str(3))

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	img1_copy = copy.deepcopy(img1)
	img2_copy = copy.deepcopy(img2)
	# img3_copy = copy.deepcopy(img3)

	N_best1 = anms(gray1, img1_copy, corners1, 50, str(1))
	N_best2 = anms(gray2, img2_copy, corners2, 50, str(2))
	# N_best3 = anms(gray3, img3_copy, corners3, 2, str(3)


	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	img1_copy = copy.deepcopy(img1)
	img2_copy = copy.deepcopy(img2)
	# img3_copy = copy.deepcopy(img3)

	descs1 = getDescriptors(gray1, img1_copy, N_best1, str(1))
	descs2 = getDescriptors(gray2, img2_copy, N_best2, str(2))
	# descs3 = getDescriptors(gray3, img3_copy, N_best3, str(3))

	# # # """
	# # # Feature Matching
	# # # Save Feature Matching output as matching.png
	# # # """

	# matched = match_features(descs1, descs2)
	img1_copy = copy.deepcopy(img1)
	img2_copy = copy.deepcopy(img2)
	# img3_copy = copy.deepcopy(img3)
	matched1 = featureMatching(descs1, descs2, 0.5)
	# print(matched)

	keypoints1 = []
	keypoints2 = []
	for n1 in N_best1: 
		keypoints1.append(cv2.KeyPoint(n1[0], n1[1], 5))

	for n2 in N_best2:
		keypoints2.append(cv2.KeyPoint(n2[0], n2[1], 5))


	matchedImg = plotMatches(img1_copy, img2_copy, keypoints1, keypoints2, matched1, "FeatureMatching")


	# matches_keypoints = []
	# for match in matches:
	# 	matches_keypoints.append()
	

	

	# print(len(matched))

	# # """
	# # Refine: RANSAC, Estimate Homography
	# # """

	img1_copy = copy.deepcopy(img1)
	img2_copy = copy.deepcopy(img2)
	# img3_copy = copy.deepcopy(img3)

	matched2, inliers1, inliers2 = RANSAC(img1_copy, img2_copy, matched1, N_best1, N_best2, 10, 500)

	matchedImg = plotMatches(img1_copy, img2_copy, keypoints1, keypoints2, matched2, "RANSAC")

	H = computeHomography(inliers1, inliers2)

	# print(H)

	# H_ = H/H[2][2]

	# print(H_)

	r, c = gray1.shape
	img_cor = [[0, c-1, 0, c-1],
			   [0, 0, r-1, r-1], 
			   [1, 1, 1, 1]]

	proj_cor = np.matmul(H, img_cor)

	for j in range(4):
		proj_cor[:,j] = proj_cor[:,j]/proj_cor[2][j]

	print(proj_cor)

	x_min = min(proj_cor[0,:])
	y_min = min(proj_cor[1,:])
	x_max = max(proj_cor[0,:])
	y_max = max(proj_cor[1,:])

	I_mat = [[1, 0, -x_min],
			 [0, 1, -y_min],
			 [0, 0, 1]]

	print(x_min)
	print(y_min)

	H_new = np.matmul(I_mat, H)

	# img_pad = np.pad(gray1, ((y_trans, 0),(x_trans, 0)), mode = 'constant', constant_values = (0, 0))
	# print(img_pad.shape)

	# cv2.imshow('dst',img_pad)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	warp_img = cv2.warpPerspective(img1_copy, H_new, (int(x_max - x_min), int(y_max-y_min)))

	cv2.imshow('dst',warp_img)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()


	# dmatchvec = []
	# for m in matched:
	# 	dmatchvec.append(cv2.DMatch(m[0], m[1], 1))

	# matchesImg1 = cv2.drawMatches(img1_copy, keypoints1, img2_copy, keypoints2, dmatchvec, matchesImg)

	# cv2.imshow('dst',matchesImg1)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

	# print(inliers1)
	pt1 = inliers1[0]
	pt2 = inliers2[0]

	cv2.circle(img1, pt1, 3, 255, -1)
	cv2.circle(img2, pt2, 3, 255, -1)

	cv2.imshow('dst',img1)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

	cv2.imshow('dst',img2)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

	# print(pt1)
	print(pt2)

	pt1_ = getProjection(H_new, pt1)
	N_best1_ = []
	for i in N_best1:
		pt11_ = getProjection(H_new, i)
		N_best1_.append((int(pt11_[0]), int(pt11_[1])))

	keypoints11 = []
	for n2 in N_best1_:
		keypoints11.append(cv2.KeyPoint(n2[0], n2[1], 5))

	matchedImg = plotMatches(warp_img, img2_copy, keypoints11, keypoints2, matched2, "RANDOM")

	print("aaaaaaaaaaa")
	print(pt1_)

	cv2.circle(warp_img, (int(pt1_[0]), int(pt1_[1])), 3, 255, -1)

	cv2.imshow('dst',warp_img)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

	r, c, _ = img2_copy.shape
	r_new, c_new, _ = warp_img.shape

	xl, xr, yu, yb = getPadding(warp_img, img1_copy, pt2, pt1_)

	output = np.pad(warp_img, ((max(0,yu),abs(min(0,yb))),(max(0,xl),abs(min(0,xr))),(0,0)), mode = 'constant', constant_values = 0)
	print("Image size: ")
	print(warp_img.shape)
	print(output.shape)

	cv2.imshow('dst',output)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

	print("upper start point: " + str(abs(min(yu,0))))
	print("lower end point: " + str(abs(min(yu,0))+r))

	output[abs(min(yu,0)):abs(min(yu,0))+r, abs(min(xl,0)):abs(min(xl,0))+c] = img2

	cv2.imwrite('Images/output.jpg', output)

	cv2.imshow('dst',output)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

	# print(r, c)
	# print(r_new, c_new)

	# # print(pt1_)
	# # print(pt2)

	# xt1 = int(round(pt1_[0] - pt2[0]))
	# yt1 = int(round(pt1_[1] - pt2[1]))

	# print("xt1: " + str(xt1))
	# print("yt1: " + str(yt1))

	# xt = c + xt1 - c_new
	# yt = r + yt1 - r_new

	# print("Why?????????????")
	# print(xt)
	# print(yt)

	# output = np.pad(warp_img, ((0,yt),(0,0),(0,0)), mode = 'constant', constant_values = 0)
	# print(warp_img.shape)
	# print(output.shape)

	# cv2.imshow('dst',output)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	# output[yt1:output.shape[0], xt1:c+xt1] = img2

	# cv2.imwrite('Images/output2.jpg', output)

	# cv2.imshow('dst',output)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()



    
if __name__ == '__main__':
    main()
 
