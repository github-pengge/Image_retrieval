# -*- coding: utf-8 -*-
from scipy.cluster.vq import *
import numpy as np 
import os
import pickle
from hierarchical_kmeans import hierarchical_kmeans as hkmeans, tree


class Vocabulary(object):
	def __init__(self, name):
		self.name = name
		self.idf = {}
		self.hk = None

	def __len__(self):
		return len(self.idf)

	def save(self):
		print('Saving vocabulary...')
		filename = 'vocabulary-%s-%s.idx' % (self.name, self.hk.total_k)
		with open(filename, 'wb') as f:
			pickle.dump([self.hk, self.idf], f)
		print('Done!')

	@staticmethod
	def load(vocabulary_file):
		print('Loading vocabulary from %s' % vocabulary_file)
		if '/' in vocabulary_file:
			tmp = vocabulary_file.split('/')[-1].split('-')
		elif '\\' in vocabulary_file:
			tmp = vocabulary_file.split('\\')[-1].split('-')
		else:
			tmp = vocabulary_file.split('-')
		if len(tmp) != 3:
			voc_name = None
		elif tmp[0] == 'vocabulary':
			voc_name = tmp[1]
			voc_len = int(tmp[2].split('.')[0])
		else:
			print('Vocabulary file may not be created by this code.')
		vocabulary = Vocabulary(voc_name)
		with open(vocabulary_file, 'rb') as f:
			[vocabulary.hk, vocabulary.idf] = pickle.load(f)
		assert vocabulary.hk.total_k == len(vocabulary.idf), 'File %s is not a standard vocabulary file.' % vocabulary_file
		if 'voc_len' in dir():
			if len(vocabulary) != voc_len:
				print('Warning: vocabulary size(%s) does not match with what file name claimed(%s).' % (len(vocabulary), voc_len))
		print('Done!')
		return vocabulary

	def train(self, feature_lists, clusters, subsampling=-1, iter=1):
		print('Creating code book(vocabulary)...')
		n_image = len(feature_lists)
		if n_image == 0:
			print('Empty feature lists!')
			return
		descriptors = np.asarray(feature_lists[0])
		des2im_id = np.array([0] * len(feature_lists[0]))
		for i in range(1, n_image):
			if feature_lists[i] is None or feature_lists[i].shape[0] == 0:
				continue
			if subsampling > 0:
				descriptors = np.vstack((descriptors, feature_lists[i][::subsampling, :]))
				des2im_id = np.concatenate((des2im_id, np.array([i] * len(feature_lists[i][::subsampling]))))
			else:
				descriptors = np.vstack((descriptors, feature_lists[i]))
				des2im_id = np.concatenate((des2im_id, np.array([i] * len(feature_lists[i]))))
		N = descriptors.shape[0]
		# self.des2im_id = des2im_id

		self.hk = hkmeans(clusters)
		self.hk.cluster(descriptors, des2im_id, iter)

		leaves = []
		self.hk.root.gather_leaves(leaves)
		leaves_with_imlist = []
		self.hk.root.gather_data_from_leaves(leaves_with_imlist, gather_additional_data=True)
		for i, leaf in enumerate(leaves):
			self.idf[leaf] = np.log(N/(len(leaves_with_imlist[i])+1))
		

	def sims(self, descriptor):
		im_sims = {}
		if descriptor is None:
			return []
		if len(descriptor) == 0:
			return []
		for des in descriptor:
			node = self.hk.find_cluster(des)
			for im_id in node.get_additional_data():
				im_sims[im_id[0]] = im_sims.get(im_id[0], im_id[1]) * self.idf[node]
		im_sims = list(im_sims.items())
		im_sims.sort(key=lambda x: x[1], reverse=True)
		im_most_sims = [_im[0] for _im in im_sims]
		return im_most_sims


def construct_vocabulary(descriptors, clusters, vocabulary_name, subsampling=-1):
	'''
	Construct codebook by hierachical k-means(k-means tree)
	Input:
		descriptors: list object, each element is sift descriptors of an image, 
				you should make sure that it is organized in the order of image ids' order,
				simply that order can be ASCE order of images' name
		clusters: a tree structure object created by class `tree`, it shows how to do the 
				hierachical k-means, for example:
				clusters = tree('root')
				for i in range(100):
					x = tree('l1-c%d' % (i+1))
					clusters.add_child(x)
					for j in range(50):
						y = tree('l2-c%d-c%d' % (i+1, j+1))
						x.add_child(y)
						for k in range(20):
							z = tree('l3-c%d-c%d-c%d' % (i+1, j+1, k+1))
							y.add_child(z)
				This code create a tree with 3 levels, it tells the constructer to create a
				k-means tree with depth-3: for first level, doing a 100-class k-means; for each
				branch(100 branches) in level 2, doing a 50-class k-means; and for each branch in
				level 3, doing a 20-class k-means. Note that if you need a 100-class-clustering,
				k-means will not always get the exactly 100 classes, some class may be empty, so 
				commonly speaking, the example given above will get a total clustering of less than
				100*50*20 classes, but I promise this is not a problem.
		vocabulary_name: vocabulary's name
		subsampling: int, default -1(not subsampling), since doing k-means can be very slow,
				subsampling can accelerate up, but performance will decline too. 
	Return:
		vocabulary: instance of Vocabulary
	'''
	vocabulary = Vocabulary(vocabulary_name)
	vocabulary.train(descriptors, clusters, subsampling=subsampling)
	vocabulary.save()
	return vocabulary
	

def test():
	print('Running test on random data...')
	feature_lists = [np.random.randn(50, 128) for _ in range(np.random.randint(low=100, high=200))]
	vocabulary = Vocabulary('random2')

	clusters = tree('root', 0)
	x = tree('l1-c1@1', 1)
	y = tree('l1-c2@2', 2)
	z = tree('l1-c3@3', 3)
	clusters.add_child(x)
	clusters.add_child(y)
	clusters.add_child(z)
	z = tree('l2-x@1', 3)
	w = tree('l2-x@2', 2)
	p1 = tree('l2-y@1', 2)
	p2 = tree('l2-y@2', 2)
	p3 = tree('l2-y@3', 2)
	x.add_children([z,w])
	y.add_children([p1,p2,p3])

	vocabulary.train(feature_lists, clusters)
	vocabulary.save()
	print(vocabulary.idf)
	leaves_with_imlist = []
	vocabulary.hk.root.gather_data_from_leaves(leaves_with_imlist, gather_additional_data=True)
	print(leaves_with_imlist)
	print('Passed.')


if __name__ == '__main__':
	test()
