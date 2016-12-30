# -*- coding: utf-8 -*-
# Note that this file can only run under python2, due to the opencv things...
import os
import cv2
import pickle
import numpy as np


def sift_detect_and_compute(images, normalize=False, return_keypoints=False, keep_top_k=-1):
	'''
	Compute sift on a list of images
	Input:
		images: list, each element is a numpy.ndarray, an image reading by opencv, to conpute a single image's 
				sift, please use [a_image] as the input
		normalize: True for doing L2 normalization on each sift descriptors
		return_keypoints: whether to return keypoints or not
		keep_top_k: int, default -1(keep all descriptors), keeping *keep_top_k* top-response sift descriptors
					(if *keep_top_k* descriptors are available)
	Return:
		descriptors: list, each element of list is sift descriptors of an image
		keypoints: (if return_keypoints=True, this will return) list, each element of list is keypoints of an image
	'''
	print('Sift detecting...')
	keypoints = []
	descriptors = []
	n_image = len(images)
	if cv2.__version__.split('.')[0] == '3':
		s = cv2.xfeatures2d.SIFT_create()
	else:
		s = cv2.SIFT()
	for i in range(n_image):
		kp = s.detect(images[i])
		kp, des = s.compute(images[i], kp)
		if keep_top_k > 0:
			order = np.argsort([-k.response for k in kp])
			kp = [kp[order[j]] for j in range(min(keep_top_k, len(kp)))]
			des = np.array([des[order[j]] for j in range(min(keep_top_k, len(kp)))])
		keypoints.append(kp)
		if des is None or des.shape[0] == 0:
			des = np.zeros((0, 128))
		if normalize:
			des = des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-15)
		descriptors.append(des)
	print('Done!')
	if return_keypoints:
		return descriptors, keypoints
	else:
		return descriptors


def surf_detect_and_compute(images, normalize=False, return_keypoints=False, keep_top_k=-1):
	'''
	Compute surf on a list of images
	Input:
		images: list, each element is a numpy.ndarray, an image reading by opencv
		normalize: True for doing L2 normalization on each surf descriptors
		return_keypoints: whether to return keypoints or not
		keep_top_k: int, default -1(keep all descriptors), keeping *keep_top_k* top-response surf descriptors
	Return:
		descriptors: list, each element of list is surf descriptors of an image
		keypoints: (if return_keypoints=True, this will return) list, each element of list is keypoints of an image
	'''
	print('Surf detecting...')
	keypoints = []
	descriptors = []
	n_image = len(images)
	if cv2.__version__.split('.')[0] == '3':
		s = cv2.xfeatures2d.SURF_create()
	else:
		s = cv2.SURF()
	for i in range(n_image):
		kp = s.detect(images[i])
		kp, des = s.compute(images[i], kp)
		if keep_top_k > 0:
			order = np.argsort([-k.response for k in kp])
			kp = [kp[order[j]] for j in range(min(keep_top_k, len(kp)))]
			des = np.array([des[order[j]] for j in range(min(keep_top_k, len(kp)))])
		keypoints.append(kp)
		if des is None or des.shape[0] == 0:
			des = np.zeros((0, 64))
		if normalize:
			des = des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-15)
		descriptors.append(des)
	print('Done!')
	if return_keypoints:
		return descriptors, keypoints
	else:
		return descriptors


def brisk_detect_and_compute(images, normalize=False, return_keypoints=False, keep_top_k=-1):
	'''
	Compute brisk feature on a list of images
	Input:
		images: list, each element is a numpy.ndarray, an image reading by opencv
		normalize: True for doing L2 normalization on each brisk descriptors
		return_keypoints: whether to return keypoints or not
		keep_top_k: int, default -1(keep all descriptors), keeping *keep_top_k* top-response brisk descriptors
	Return:
		descriptors: list, each element of list is brisk descriptors of an image
		keypoints: (if return_keypoints=True, this will return) list, each element of list is keypoints of an image
	'''
	print('Brisk detecting...')
	keypoints = []
	descriptors = []
	n_image = len(images)
	if cv2.__version__.split('.')[0] == '3':
		s = cv2.BRISK_create()
	else:
		s = cv2.BRISK()
	for i in range(n_image):
		kp = s.detect(images[i])
		kp, des = s.compute(images[i], kp)
		if keep_top_k > 0:
			order = np.argsort([-k.response for k in kp])
			kp = [kp[order[j]] for j in range(min(keep_top_k, len(kp)))]
			des = np.array([des[order[j]] for j in range(min(keep_top_k, len(kp)))])
		keypoints.append(kp)
		if normalize:
			des = des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-15)
		descriptors.append(des)
	print('Done!')
	if return_keypoints:
		return descriptors, keypoints
	else:
		return descriptors


def vlad_compute(centroids, descriptors, normalize=True):
	'''Compute VLAD feature for images' descriptors, e.g. SIFT/SURF...
	'descriptors' is a list, each element is an image's descriptors.
	'''
	vlad_feat = []
	for des in descriptors:
		vlad_feat.append(vlad(centroids, des, normalize))
	return np.array(vlad_feat)


def vlad(centroids, des, normalize=True):
	'''Compute VLAD feature for an image's descriptors'''
	centroids = np.array(centroids)
	des = np.array(des)
	if des is None or des.shape[0] == 0:
		des = np.ones(centroids.shape)
	dist = EuDist(des, centroids)
	nn = np.argmin(dist, axis=1)
	feat = np.zeros(centroids.shape)
	for i in range(des.shape[0]):
		feat[nn[i]] += des[i] - centroids[nn[i]]
	feat = feat.reshape((feat.shape[0]*feat.shape[1]))
	norm = np.linalg.norm(feat)
	if norm == 0:
		feat = np.ones(centroids.shape)
	if normalize:
		feat = feat / norm
	return feat


def EuDist(X, Y=None, bSqrt=True, bRow=True): # An accelerated function
	'''
	Compute Euclidean distances between each data of X and each data of Y.
	:param X: data in row
	:param Y: data in row, if Y == None, then return EuDist(X,X,bSqrt)
	:param bSqrt: whether to return ||x-y||^2 or ||x-y||
	:param bRow: whether each row is a data
	:return: dist: the Euclidean distance of each x in X and each y in Y.
					Each row is the distance of x to all y in Y.
	@author: Gapeng
	'''
	if(not bRow):
		X = X.T
		Y = Y.T
	if(type(Y) == type(None) or len(Y) == 0):
		n       = X.shape[0]
		XX      = np.atleast_2d(np.sum(X * X, axis=1))
		if(len(X.shape) == 1):
			XY  = np.dot(np.atleast_2d(X).T,np.atleast_2d(X))
		else:
			XY  = np.dot(X,X.T)
		dist    = XX + XX.T - 2 * XY
		dist[dist < 0] = 0
		if(bSqrt):
			dist = np.sqrt(dist)
		# dist    = np.max(np.vstack((dist.flatten(),dist.T.flatten())),axis=0)
		# dist    = np.reshape(dist,newshape=(n,n))
	else:
		if(not bRow):
			Y = Y.T
		if(len(X.shape) < 2):
			X = np.atleast_2d(X)
		if(len(Y.shape) < 2):
			Y = np.atleast_2d(Y)
		n       = X.shape[0]
		m       = Y.shape[0]
		XX      = np.sum(X * X,axis=1,keepdims=True)
		YY      = np.sum(Y * Y,axis=1,keepdims=True)
		if(len(X.shape) == 1):
			XY  = np.dot(np.atleast_2d(X).T,np.atleast_2d(Y))
		else:
			XY  = np.dot(X,Y.T)
		dist    = XX + YY.T - 2 * XY
		dist[dist < 0] = 0
		if(bSqrt):
			dist = np.sqrt(dist)
		# dist    = np.max(np.vstack((dist.flatten(),dist.T.flatten())),axis=0)
		# dist    = np.reshape(dist,newshape=(n,m))
		if(m < 2):
			dist= dist[:,0]
	if(not bRow):
		dist    = dist.T
	return dist


def extract_lbp_hist(images, radius=3, n_points=50, method='uniform'):
	lbp_hist = []
	n_image = len(images)
	for image in images:
		lbp = local_binary_pattern(image, n_points, radius, method)
		lbp = local_binary_pattern(image, n_points, radius, method=method) 
		x = itemfreq(lbp.ravel())
		hist = x[:,1]/sum(x[:,1])
		lbp_hist.append(hist)
	return lbp_hist


