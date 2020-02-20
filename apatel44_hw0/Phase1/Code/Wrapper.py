#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import os
import numpy as np
import cv2
import sklearn.cluster
import skimage.transform

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import math

import matplotlib.pyplot as plt
from matplotlib import cm


def generate_DoG_FilterBank(scales, orientations, kernal_size):

	sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

	filter_bank = []

	filtBankDisp = np.array([])

	for i in range(1, scales+1):
		sigma = i;
		mu = 0;
		gauss_kernal = generate_gaussian(sigma, mu, kernal_size)
		DoG = cv2.filter2D(gauss_kernal,-1,sobelX)

		filtDisp = np.array([])

		for orient in range(0, orientations):
			angle=180*orient/(orientations-1)
			image_center = ((kernal_size - 1)/2, (kernal_size - 1)/2)
			rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
			rot_kernal = cv2.warpAffine(DoG, rot_mat, (kernal_size, kernal_size))
			filter_bank.append(rot_kernal)

			if filtDisp.size == 0:
				filtDisp = rot_kernal
			else:
				filtDisp = cv2.hconcat([filtDisp, rot_kernal])

		if filtBankDisp.size == 0:
			filtBankDisp = filtDisp
		else:
			filtBankDisp = cv2.vconcat([filtBankDisp, filtDisp])

	# plt.imshow(filtBankDisp, cmap = 'gray')
	# plt.show()
	plt.imsave('FilterBank/DoG Filter Bank', filtBankDisp, cmap = 'gray')

	for k in range(len(filter_bank)):
		plt.subplot(4,8,k+1)
		plt.axis('off')
		plt.imshow(filter_bank[k],cmap='gray')
	# plt.show()
	plt.savefig('FilterBank/DoG Filter Bank 2')
	plt.close()

	return filter_bank

def generate_gaussian(sigma, mu, size):
	x, y = np.meshgrid(np.linspace(-7,7,size), np.linspace(-7,7,size))
	d = np.sqrt(x*x+y*y)
	result = np.exp(-((d-mu)**2 / ( 2.0 * sigma**2 )))
	return result


def gaussian1d(sigma, mean, x, ord):
	x = np.array(x)
	x_ = x - mean
	var = sigma**2

	# Gaussian Function
	g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

	if ord == 0:
		g = g1
		return g
	elif ord == 1:
		g = -g1*((x_)/(var))
		return g
	else:
		g = g1*(((x_*x_) - var)/(var**2))
		return g

def gaussian2d(sup, scales):
	var = scales * scales
	shape = (sup,sup)
	n,m = [(i - 1)/2 for i in shape]
	x,y = np.ogrid[-m:m+1,-n:n+1]
	g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
	return g

def log2d(sup, scales):
	var = scales * scales
	shape = (sup,sup)
	n,m = [(i - 1)/2 for i in shape]
	x,y = np.ogrid[-m:m+1,-n:n+1]
	g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
	h = g*((x*x + y*y) - var)/(var**2)
	return h

def makefilter(scale, phasex, phasey, pts, sup):

	gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
	gy = gaussian1d(scale,   0, pts[1,...], phasey)

	image = gx*gy

	image = np.reshape(image,(sup,sup))
	return image


def gabor(filt_size, Lambda, theta, phi, sigma, gamma):

	x, y = np.meshgrid(np.linspace(-7, 7, filt_size[0]), np.linspace(-7, 7, filt_size[1]))

	x_ = x * np.cos(theta) + y * np.sin(theta)
	y_ = -x * np.sin(theta) + y * np.cos(theta)

	gaussian = np.exp(-(x_**2 + (gamma**2)*(y_**2))/(2*sigma**2))
	sinusoidal = np.cos((2*np.pi*x_)/Lambda + phi)

	gf = gaussian * sinusoidal;

	return gf;


def getFilterResponse(filt_bank, im):
	nFilt = len(filt_bank)
	# print("Inside Function: ")
	# print("Filter size ")
	# print(nFilt)

	filt_resp = np.array(im)

	

	for i in range(0,nFilt):
		# print(type(f))
		# cv2.imshow("xxxx", f)
		# cv2.waitKey(0)
		f = filt_bank[i]
		resp = cv2.filter2D(im,-1,f)
		filt_resp = np.dstack([filt_resp, resp])

	return filt_resp

def half_disk(radius):
    a=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    mask2 = x*x + y*y <= radius**2
    a[mask2] = 0
    b=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    p = x>-1
    q = y>-radius-1
    mask3 = p*q
    b[mask3] = 0

    return a, b

def disk_masks(scales, orients):
	flt = list()
	orients = np.linspace(0,360,orients)
	for i in scales:
		r = i
		g = list()
		a,b = half_disk(radius = r)

		for i,orient in enumerate(orients):
			c1 = skimage.transform.rotate(b, orient, cval =1)
			z1 = np.logical_or(a,c1)
			z1 = z1.astype(np.int)
			b2 = np.flip(b,1)
			
			c2 = skimage.transform.rotate(b2, orient, cval =1)
			z2 = np.logical_or(a,c2)
			z2 = z2.astype(np.int)
			
			flt.append(z1)
			flt.append(z2)

	return flt

def plot_halfdisks(masks):
	l = len(masks)
	plt.subplots(l/5,5,figsize=(20,20))
	for i in range(l):
		plt.subplot(l/4,4,i+1)
		plt.axis('off')
		plt.imshow(masks[i],cmap='binary')

	plt.savefig('Half-Disks.png')
	plt.close()


def chi_sqr_gradient(Img, bins,filter1,filter2):
	chi_sqr_dist = Img*0
	g = list()
	h = list()
	for i in range(bins):
		#numpy.ma.masked_where(condition, a, copy=True)[source]
		#Mask an array where a condition is met.
		img = np.ma.masked_where(Img == i,Img)
		img = img.mask.astype(np.int)
		g = cv2.filter2D(img,-1,filter1)
		h = cv2.filter2D(img,-1,filter2)
		chi_sqr_dist = chi_sqr_dist + ((g-h)**2 /(g+h))
	return chi_sqr_dist/2

def gradient(Img, bins, filter_bank):
	gradVar = Img
	for N in range(len(filter_bank)/2):
		g = chi_sqr_gradient(Img, bins, filter_bank[2*N],filter_bank[2*N+1])
		gradVar = np.dstack((gradVar,g))
	mean = np.mean(gradVar,axis =2)
	return mean





def main():


	# Load the images and save them in an array
	im = plt.imread('/home/akanksha/Documents/CMSC733/apatel44_hw0/Phase1/BSDS500/SobelBaseline/1.png');
	print(im.shape)

	current_dir = os.path.dirname(os.path.abspath(__file__))
	temp_path, dir_name = os.path.split(current_dir)
	image_dir = os.path.join(temp_path, "BSDS500/Images")

	images = []
	image_names = []

	for name in os.listdir(image_dir):
		print(name)
		im = cv2.imread(os.path.join(image_dir, name));
		if im is not None:
			images.append(im)
			image_names.append(name)
		else:
			print("None")


	
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	DoG_scales = 2
	DoG_orient = 16
	DoG_kernal_size = 11

	# ToDo: Create new folder
	filter_bank = []
	DoG_bank = []

	DoG_bank = generate_DoG_FilterBank(DoG_scales, DoG_orient, DoG_kernal_size)
	filter_bank = DoG_bank
	print("DoG Filter Bank created")

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	LM_kernal_size = 49
	scalex  = np.sqrt(2) * np.array([1,2,3])
	norient = 6
	nrotinv = 12

	nbar  = len(scalex)*norient
	nedge = len(scalex)*norient
	nf    = nbar+nedge+nrotinv
	F     = np.zeros([LM_kernal_size,LM_kernal_size,nf])
	hsup  = (LM_kernal_size - 1)/2

	x = [np.arange(-hsup,hsup+1)]
	y = [np.arange(-hsup,hsup+1)]

	[x,y] = np.meshgrid(x,y)

	orgpts = [x.flatten(), y.flatten()]
	orgpts = np.array(orgpts)

	count = 0
	for scale in range(len(scalex)):
		for orient in range(norient):
			angle = (np.pi * orient)/norient
			c = np.cos(angle)
			s = np.sin(angle)
			rotpts = [[c+0,-s+0],[s+0,c+0]]
			rotpts = np.array(rotpts)
			rotpts = np.dot(rotpts,orgpts)
			F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, LM_kernal_size)
			F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, LM_kernal_size)
			count = count + 1

	count = nbar+nedge
	scales = np.sqrt(2) * np.array([1,2,3,4])

	for i in range(len(scales)):
		F[:,:,count]   = gaussian2d(LM_kernal_size, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:,:,count] = log2d(LM_kernal_size, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:,:,count] = log2d(LM_kernal_size, 3*scales[i])
		count = count + 1

	p,q,r = F.shape

	filtBankDisp = np.array([])

	for k in range(r):
		plt.subplot(4,12,k+1)
		plt.axis('off')
		plt.imshow(F[:,:,k],cmap='gray')
	# plt.show()
	plt.savefig('FilterBank/LM Filter Bank')
	plt.close()
	print("LM Filter Bank created")

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

	filt_size = (49, 49)
	filt_size2 = (25, 25)
	sigma = [1, 2, 3, 4]
	nOrient = 8

	filtBankDisp = np.array([])

	gf = gabor(filt_size, 1, 0, 0, sigma[i], 1)	

	Gabor_bank = []

	for i in range(0, 4):
		gf = gabor(filt_size, sigma[i], 0, 0, sigma[i], 1)
		filtDisp = np.array([])
		for orient in range(0, nOrient):
			angle=180*orient/(nOrient)
			image_center = ((filt_size[0] - 1)/2, (filt_size[1] - 1)/2)
			rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
			rot_kernal = cv2.warpAffine(gf, rot_mat, filt_size)
			rot_kernal = rot_kernal[:, 11:36]
			rot_kernal = rot_kernal[11:36, :]
			# print(rot_kernal.shape)
			Gabor_bank.append(rot_kernal)
			filter_bank.append(rot_kernal)

			if filtDisp.size == 0:
				filtDisp = rot_kernal
			else:
				filtDisp = cv2.hconcat([filtDisp, rot_kernal])

		if filtBankDisp.size == 0:
			filtBankDisp = filtDisp
		else:
			filtBankDisp = cv2.vconcat([filtBankDisp, filtDisp])

	# plt.imshow(filtBankDisp, cmap = 'gray')
	# plt.show()
	plt.imsave('FilterBank/Gabor_Filter_Bank', filtBankDisp, cmap = 'gray')
	print("Gabor Filter Bank created")
	

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	c = disk_masks([5,7,16], 8)
	print("Half Disks created")

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""

	# for i in range(3,4):
	for i in range(len(images)):
		im = plt.imread('/home/akanksha/Documents/CMSC733/apatel44_hw0/Phase1/BSDS500/Images/' + str(i+1) + '.jpg')
		name = str(i)
		
		filt_resp = getFilterResponse(filter_bank, im)

		p, q, r = filt_resp.shape
		inp = np.reshape(filt_resp,((p*q),r))

		kmeans = sklearn.cluster.KMeans(n_clusters = 64, random_state = 2)
		kmeans.fit(inp)
		labels = kmeans.predict(inp)
		l = np.reshape(labels,(p,q))
		
		np.save('TextonMap/Texton' + name+ '.npy', l)
		# l = np.load('TextonMap/Texton' + name+ '.npy')
		plt.imsave('TextonMap/Texton' + name, l)
		plt.close()
		print("Texton Map created for " + name)

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""




		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		Tg = gradient(l, 64, c)
		np.save('TextonGradient/Tg' + name +'.npy', Tg)
		# Tg = np.load('TextonGradient/Tg' + name +'.npy')
		plt.imsave('TextonGradient/Texton_Gradient' + name, Tg)
		plt.close()
		print("Texton Gradient created for " + name)

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""

		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		p, q = gray.shape;
		inp = np.reshape(gray,((p*q),1))

		kmeans = sklearn.cluster.KMeans(n_clusters = 16, random_state = 2)
		kmeans.fit(inp)
		labels = kmeans.predict(inp)
		l = np.reshape(labels,(p,q))
		
		np.save('BrightnessMap/BrightnessMap' + name + '.npy', l)
		# l = np.load('BrightnessMap/BrightnessMap' + name + '.npy')
		plt.imsave('BrightnessMap/Brightness_Map' + name, l)
		plt.close()
		print("Brightness Map created for " + name)

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		Bg = gradient(l, 16, c)
		plt.imshow(Bg)
		np.save('BrightnessGradient/BrightnessGradient' + name + '.npy', Bg)
		# Bg = np.load('BrightnessGradient/BrightnessGradient' + name + '.npy')
		plt.imsave('BrightnessGradient/Brightness_Gradient' + name, Bg)
		plt.close()
		print("Brightness Gradient created for " + name)

		"""
		Generate Color Map
		Perform color binning or clustering
		"""

		p, q, r= im.shape;
		inp = np.reshape(im,((p*q),r))
		
		kmeans = sklearn.cluster.KMeans(n_clusters = 16, random_state = 2)
		kmeans.fit(inp)
		labels = kmeans.predict(inp)
		l = np.reshape(labels,(p,q))
		
		np.save('ColorMap/ColorMap' + name + '.npy', l)
		l = np.load('ColorMap/ColorMap' + name + '.npy')
		plt.imsave('ColorMap/Color_Map' + name, l)
		plt.close()
		print("Color Map created for " + name)


		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		Cg = gradient(l, 16, c)
		np.save('ColorGradient/ColorGradient' + name + '.npy', Cg)
		Cg = np.load('ColorGradient/ColorGradient' + name + '.npy')
		plt.imsave('ColorGradient/Color_Gradient' + name, Cg)
		plt.close()	
		print("Color Gradient created for " + name)


		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""

		sobel = plt.imread('/home/akanksha/Documents/CMSC733/apatel44_hw0/Phase1/BSDS500/SobelBaseline/' + str(i+1) + '.png');
		# sobel = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		canny = plt.imread('/home/akanksha/Documents/CMSC733/apatel44_hw0/Phase1/BSDS500/CannyBaseline/' + str(i+1) + '.png');
		# canny = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		tbg = (Tg + Bg + Cg)/3
		plt.imsave('tbg', tbg, cmap = 'gray')
		cs = (0.7*sobel + 0.3*canny)
		plt.imsave('cs', cs, cmap = 'gray')
		pb_edge =  np.multiply(tbg, cs)

		plt.imshow(pb_edge, cmap = 'gray')
		# plt.show()	
		plt.imsave('pbLite/pblite_image' + name, pb_edge, cmap = 'gray')
		plt.close()

		print("pbLite Image created for " + name)

		# im = cv2.imread("/home/akanksha/Documents/CMSC733/apatel44_hw0/Phase1/BSDS500/SobelBaseline/7.png")
		# sobel = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		# # for image_name in os.listdir(sobel_dir):
		# # 	print(image_name)
		# # 	im = cv2.imread(os.path.join(sobel_dir, image_name));
		# # 	if im is not None:
		# # 		sobel.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
		# # 	else:
		# # 		print("None")

		# # print(len(sobel))



		# """
		# Read Canny Baseline
		# use command "cv2.imread(...)"
		# """

		# canny_dir = os.path.join(temp_path, "BSDS500/CannyBaseline/")

		# im = cv2.imread("/home/akanksha/Documents/CMSC733/apatel44_hw0/Phase1/BSDS500/CannyBaseline/7.png")
		# canny = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


		# # for image_name in os.listdir(canny_dir):
		# # 	print(image_name)
		# # 	im = cv2.imread(os.path.join(sobel_dir, image_name));
		# # 	if im is not None:
		# # 		canny.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
		# # 	else:
		# # 		print("None")

		# # print(len(canny))


		# """
		# Combine responses to get pb-lite output
		# Display PbLite and save image as PbLite_ImageName.png
		# use command "cv2.imwrite(...)"
		# """

		# tbg = (Tg + Bg + Cg)/3
		# plt.imsave('tbg', tbg, cmap = 'gray')
		# cs = (0.75*sobel + 0.25*canny)
		# plt.imsave('cs', cs, cmap = 'gray')
		# pb_edge =  np.multiply(tbg, cs)

		# plt.imshow(pb_edge, cmap = 'gray')
		# # plt.show()	
		# plt.imsave('pblite_image', pb_edge, cmap = 'gray')
		# plt.close()



    
if __name__ == '__main__':
    main()
 


