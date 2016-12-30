# -*- encoding: utf-8 -*-
from vocabulary import Vocabulary
import os
import cv2


def load_dataset(data_path, dataset_name='ukbench', first_n=-1, return_images=True):
	dataset_name = dataset_name.lower()
	assert dataset_name in ['ukbench', 'oxbuilding', 'holiday']

	print('Loading dataset from %s' % data_path)
	image_name = sorted(os.listdir(data_path))
	if first_n > 0:
		image_name = image_name[0:first_n]
	image_path = [os.path.join(data_path, image) for image in image_name]
	image_id = []
	if return_images:
		images = []

	for i, image in enumerate(image_path):
		if dataset_name == 'ukbench':
			image_id.append(int(image_name[i][7:12]))
		elif dataset_name == 'oxbuilding':
			l = len(image_name[i])
			building = image_name[i][:l-11]
			image_id.append(building)
		else:
			label = int(image_name[i][:6]) // 100 - 1000
			image_id.append(label)

		if return_images:
			im = cv2.imread(image)
			if im is None:
				print('While reading image %s, got a error, this image was escaped.' % image)
			else:
				images.append(im)
	print('Done!')
	
	if return_images:
		return images, image_id, image_name
	else:
		return image_id, image_name

def load_input_file(input_file):
	with open(input_file, 'r') as f:
		content = f.read().strip()
		lines = content.split('\r\n')
		data_path = None
		index_path = None
		reranking_mat_path = None
		save_result_to = None
		query_list = []
		n_lines = len(lines)
		i = 0
		while(i < n_lines):
			line = lines[i]
			if 'Database path:' == line.strip():
				data_path = lines[i+1].strip()
				i += 2
				continue
			if 'Index path:' == line.strip():
				index_path = lines[i+1].strip()
				i += 2
				continue
			if 'Reranking mat path:' == line.strip():
				reranking_mat_path = lines[i+1].strip()
				i += 2
				continue
			if 'Retrieval result path:' == line.strip():
				save_result_to = lines[i+1].strip()
				i += 2
				continue
			if 'End' == line.strip():
				break
			if 'Query list:' == line.strip() or '' == line.strip():
				i += 1
				continue
			query_list.append(line.strip())
			i += 1
	return data_path, index_path, reranking_mat_path, save_result_to, query_list


def load_vocabulary(vocabulary_filename):
	voc = Vocabulary.load(vocabulary_filename)
	return voc


# def load_ukbench(data_path, first_n=-1):
# 	print('Loading UKbench dataset...')
# 	image_name = sorted(os.listdir(data_path))
# 	if first_n > 0:
# 		image_name = image_name[0:first_n]
# 	image_path = [os.path.join(data_path, image) for image in image_name]
# 	images = []
# 	image_id = [int(image[7:12]) for image in image_name]  # image id, ground truth can be calculated by floor(id/4)
# 	for image in image_path:
# 		im = cv2.imread(image)
# 		if im is None:
# 			print('While reading image %s, got a error, this image was escaped.' % image)
# 		else:
# 			images.append(im)
# 	print('Done!')
# 	return images, image_id, image_name


# def load_oxbuilding(data_path, first_n=-1):
# 	print('Loading oxbuilding dataset...')
# 	image_name = sorted(os.listdir(data_path))
# 	if first_n > 0:
# 		image_name = image_name[0:first_n]
# 	image_path = [os.path.join(data_path, image) for image in image_name]
# 	images = []
# 	image_id = []
# 	for i in range(len(image_name)):
# 		l = len(image_name[i])
# 		building = image_name[i][:l-11]
# 		image_id.append(building)
# 		im = cv2.imread(image_path[i])
# 		if im is None:
# 			print('While reading image %s, got a error, this image was escaped.' % image_path[i])
# 		else:
# 			images.append(im)
# 	print('Done!')
# 	return images, image_id, image_name


# def load_holiday(data_path, first_n=-1):
# 	print('Loading holiday dataset...')
# 	image_name = sorted(os.listdir(data_path))
# 	if first_n > 0:
# 		image_name = image_name[0:first_n]
# 	image_path = [os.path.join(data_path, image) for image in image_name]
# 	images = []
# 	image_id = []
# 	for i in range(len(image_name)):
# 		label = int(image_name[i][:6]) // 100 - 1000
# 		image_id.append(label)
# 		im = cv2.imread(image_path[i])
# 		if im is None:
# 			print('While reading image %s, got a error, this image was escaped.' % image_path[i])
# 		else:
# 			images.append(im)
# 	print('Done!')
# 	return images, image_id, image_name
