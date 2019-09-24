# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:36:21 2018

@author: megha
"""

import cv2
import numpy as np
import sys


def main():
	combine, img1, img2 = getImage()
	print("Press 'H' for help!! Press 'q' to quit:")
	k = input()
	while k != 'q':
		if k == 'h':
			n = input("Enter the variance of Guassian scale:")
			wSize = input("Enter the Window Size :")
			k = input("Enter the weight of the trace in the harris conner detector(k)[0, 0.5]:")
			threshold = input("Enter the threshold value:")
			print("Result processing...........")
			res = harris(combine, n, wSize, k, threshold)
			showWin(res)
		if k == 'f':
			res = featureVector(img1, img2)
			showWin(res)
		if k == 'b':
			res = betterLocalization(combine)
			showWin(res)
		if k == 'H':
			help()
		print("Press 'H' for help!! Press 'q' to quit:")
		k = input()


def getImage():
	if len(sys.argv) == 3:
		img1 = cv2.imread(sys.argv[1])
		img2 = cv2.imread(sys.argv[2])
		#img_bw1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		#img_bw2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		#return img_bw1
		#return img_bw2
	else:
			cp = cv2.VideoCapture(0)
			for i in range(0,15):
				rvalue1,img1 = cp.read()
				rvalue2,img2 = cp.read()
			if rvalue1 and rvalue2:
				cv2.imwrite("image_captured1.jpg", img1)
				cv2.imwrite("image_captured2.jpg", img2)
	combine = np.concatenate((img1, img2), axis=1)
	return combine, img1, img2;

def cvt2Gray(img):
	img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#img_bw = cv2.cvtColor(img_bw,cv2.COLOR_GRAY2BGR)
	cv2.imshow("Display", img_bw)
	return img_bw


def showWin(img):
	#img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#img_bw = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
	cv2.imshow("Display", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def smooth(img, n):
	ker = np.ones((n, n), np.float32)/(n * n)
	dist = cv2.filter2D(img, -1, ker)
	return dist


def harris(img, n, wSize, k, threshold):
	n = int(n)
	wSize = int(wSize)
	k = float(k)
	threshold = int(threshold)
	#img = cvt2Gray(img)
	copy = img.copy()
	rList = []
	height = img.shape[0]
	width = img.shape[1]	
	offset = int(wSize / 2)
	img = cvt2Gray(img)
	img = np.float32(img)
	img = smooth(img, n)
	dy, dx = np.gradient(img)
	Ixx = dx ** 2
	Ixy = dy * dx
	Iyy = dy ** 2

	for y in range(offset, height - offset):
			for x in range(offset, width - offset):
				wIxx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
				wIxy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				wIyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				Sxx = wIxx.sum()
				Sxy = wIxy.sum()
				Syy = wIyy.sum()
				determinant = (Sxx * Syy) - (Sxy ** 2)
				trace = Sxx + Syy
				r = determinant - k *(trace ** 2)
				rList.append([x, y, r])
				if r > threshold:
							copy.itemset((y, x, 0), 0)
							copy.itemset((y, x, 1), 0)
							copy.itemset((y, x, 2), 255)
							cv2.rectangle(copy, (x + 10, y + 10), (x - 10, y - 10), (255, 0, 0), 1)
	return copy
	
def featureVector(img1, img2):
	orb = cv2.ORB_create()# Initiating SIFT detector
	keyp1, des1 = orb.detectAndCompute(img1,None) 
	keyp2, des2 = orb.detectAndCompute(img2,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)	# creating BFMatcher object
	matches = bf.match(des1,des2)	
	matches = sorted(matches, key = lambda x:x.distance)	# Sorting in the order of their distance.
	keyp1List = []
	keyp2List = []
	for m in matches:
		(x1, y1) = keyp1[m.queryIdx].pt
		(x2, y2) = keyp2[m.trainIdx].pt
		keyp1List.append((x1, y1))
		keyp2List.append((x2, y2))
	for i in range(0, 50):
		pt1 = keyp1List[i]
		pt2 = keyp2List[i]
		cv2.putText(img1, str(i), (int(pt1[0]), int(pt1[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
		cv2.putText(img2, str(i), (int(pt2[0]), int(pt2[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
	res = np.concatenate((img1, img2), axis=1)
	return res


def betterLocalization(img):
	#gray = cvt2Gray(img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dist = cv2.cornerHarris(gray,2,3,0.04)
	dist = cv2.dilate(dist,None)
	rt, dist = cv2.threshold(dist,0.01*dist.max(),255,0)
	dist = np.uint8(dist)

	rt, labels, stats, centroids = cv2.connectedComponentsWithStats(dist)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

	res = np.hstack((centroids,corners))
	res = np.int0(res)
	img[res[:,1],res[:,0]]=[0,0,255]
	img[res[:,3],res[:,2]] = [0,255,0]
	return img


def help():
	print("'h': Estimate image gradients and apply Harris corner detection algorithm.")
	print("'b': Obtain a better localization of each corner.")
	print("'f': Compute a feature vector for each corner were detected.\n")


if __name__ == '__main__':
	main()