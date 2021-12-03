import cv2
import sys, os

directory = "../image_sets/SRx2/"

count = 0
ordering = True
for filename in sorted(os.listdir(directory)):
	print(filename)
	img = cv2.imread(os.path.join(directory,filename), cv2.IMREAD_UNCHANGED)
 
	print('Original Dimensions : ',img.shape)
 
	dim = (2048, 1024)
	# resize image
	print(filename)
	if ordering:
		# print("new name: ", filename.split('_')[-1].zfill(10))
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		# print(filename.split('_')[2].zfill(10)+".png")
		cv2.imwrite(os.path.join(directory, filename.split('_')[2].zfill(10)+".png"), resized)
	# else:
		# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		# cv2.imwrite(os.path.join(directory,filename), resized)

	# cv2.imwrite(os.path.join(directory,filename.split('_')[0].zfill(10)+".png"), resized)
