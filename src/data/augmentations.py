import torch
from torchvision import transforms
import cv2
import numpy as np 
import types
from numpy import random

class Compose(object):

	def __init__(self, transforms):
		
		self.transforms = transforms

	def __call__(self, img):
		
		for transform in self.transforms:
			img = transform(img)

		# print("image type: ", img.type())
		return img

class RandomSampleCrop(object):

	def __call__(self, img):

		im_h, im_w, im_c = image_shape

		w = random.uniform(0.5 * im_w, im_w)
		h = random.uniform(0.5 * im_h, im_h)

		# aspect ratio constraint ????

		left = random.uniform(im_w - w)
		right = random.uniform(im_h - h)

		rect = np.array([int(left), int(top), int(left+w), int(top+h)])

		img = img[ rect[1]:rect[3], rect[0]:rect[2], :]

		return img

class Rotation_pano(object):

	def __init__(self, number_angles, start_degree):
		self.number_angles = number_angles
		self.start_degree = start_degree


	def __call__(self, img):

		segment = int(img.shape[1]/(self.number_angles))
		imgs = []
		for i in range(self.number_angles):
			# print("i in loop: ", i)
			a_img = np.roll(img, i*segment+self.start_degree, axis=1)
			imgs.append(a_img)

		# print( imgs[random.randint(0, self.number_angles-1)])
		return imgs[random.randint(0, self.number_angles-1)]

class Resize(object):

	def __init__(self, width, height):
		self.width =  width
		self.height = height

	def __call__(self, img):

		return cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_CUBIC)

class Split_Part(object):
	
	def __call__(self, img):

		im_c, im_h, im_w = img.shape

		w = int(im_w / 2)
		h = int(im_h / 2)

		# im_parts = [(0,w,0,h),(w,w*2,0,h), (0,w,h,h*2), (w,w*2,h,h*2)]

		im_parts = [(0, w),(w, w*2)]
	
		selected_part = im_parts[random.randint(0,1)] 
		print("image shape: ", img.shape)
		return img[:, :, selected_part[0]:selected_part[1]]


class panorama_augmentation(object):

	def __init__(self, number_of_angles, start_degree, width, height,mode):

		# print("number of angles: ", number_of_angles)
		# print("start_degree: ", start_degree)

		if mode == "train":
			self.augment = Compose([
				# RandomSampleCrop(),
				Rotation_pano(number_of_angles, start_degree),
				# Split_Part(spli),
				# Resize(width, height)
				])
		elif mode == "test":
			self.augment = Compose([
				# RandomSampleCrop(),
				# Rotation_pano(number_of_angles, start_degree),
				# Split_Part(),
				# Resize(width, height)
				])


	def __call__(self, img):
		# print("img: ", img.type())
		return self.augment(img)

