import json
import cv2 
import numpy as np

# directory = "../image_sets/images2_2"

def load_frames(start_frame_id, offset, directory):

	frame_id = str(start_frame_id+offset).zfill(6)
	# print(directory+"/"+frame_id+"jpg")
	target_im = cv2.imread(directory+"/"+frame_id+".jpg")

	source_imgs = []

	for i in range(start_frame_id, start_frame_id+offset):
		print(i)
		frame_id = str(i).zfill(6)
		im = cv2.imread(directory+"/"+frame_id+".jpg")

		source_imgs.append((i,im))

	# UV_map = cv2.imread(directory+"/UV_map.png")
	UV_map = None
	return source_imgs, target_im, UV_map

def get_all_keys(frames_dict):

	dict_0 = frames_dict['0']
	dict_1 = frames_dict['1']
	dict_2 = frames_dict['2']
	dict_3 = frames_dict['3']
	dict_4 = frames_dict['4']

	keys = []

	for k in dict_0.keys():
		if k not in keys:
			keys.append(k)

	for k in dict_1.keys():
		if k not in keys:
			keys.append(k)

	for k in dict_2.keys():
		if k not in keys:
			keys.append(k)

	for k in dict_3.keys():
		if k not in keys:
			keys.append(k)

	for k in dict_4.keys():
		if k not in keys:
			keys.append(k)

	return keys

def create_consistency_image(start_frame_id, offset, directory):

	# Create consistency images
	consistency_im = np.zeros((1280, 2560, 3)).astype(np.uint8)

	# Open dictionaries
	with open('UV2panos_dict.json', 'r') as f:
  		UV2p_dict = json.load(f)

	with open('frames_dict.json', 'r') as f:
  		frames_dict = json.load(f)



	source_imgs, target_im, UV_map = load_frames(start_frame_id, offset, directory)

	# Get current frame dictionary curent_frame_dict[frame_xy_coordinates] = [UV_xy_coordinates]
	current_frame_dict = get_dict_panorama(start_frame_id+offset, frames_dict)


	for k in current_frame_dict.keys():
		
		# print('k: ', k)
		colors = []
		
		for UV_coord in current_frame_dict[k]:
			print("UV coord: ", UV_coord)
			# Get colors based on the previus frames and their contribution on this UV_xy_
			colors.append(get_color_list(UV_coord, UV2p_dict, start_frame_id+offset, source_imgs))

		print("End")
		final_color =  np.mean(np.array(colors), axis=0)	
		# print(final_color)
		x,y = k.split(',')
		# print(x,y)

		consistency_im[int(y),int(x),:] = final_color 

	# Display consistency Image
	# im = cv2.resize(consistency_im, (1024, 512), interpolation = cv2.INTER_CUBIC)
	# print(im)
	# print(im.shape)
	cv2.imshow("Consistency Image", consistency_im)
	cv2.waitKey(0)

def softmax(x):

	"""Compute softmax values for each sets of scores in x."""
	
	e_x = np.exp(x - np.max(x))

	return e_x / e_x.sum()

def get_color_list(UV_coord, UV2p_dict, index, source_imgs):



	try:
		UV_current_pixel = UV2p_dict[UV_coord+'\n']
	except KeyError:
		# print("keys error")
		return(np.array([255,255,255]))
	
	colors_list = []
	dist_list = []
	for k in UV_current_pixel.keys():
		if int(k) != index:
			# get previous im pixel that contribute to that specific UV_location
			fr_list = UV_current_pixel[k][0].split(',')
			# print("fr_list: ", fr_list)
			y = int(fr_list[0])
			x = int(fr_list[1])
			dist = -float(fr_list[2])
			dist_list.append(dist)
			img = source_imgs[int(k)][1]

			# print(source_imgs[int(k)][0], source_imgs[int(k)][1].shape)
			# print("record: ", int(x),int(y),float(dist), img[x,y])
			# print(img[x,y])

			# cv2.imshow("Test", img)
			# cv2.waitKey(0)
			# print(x,y)
			colors_list.append( np.array(img[x][y]))

	# print("len color list: ", colors_list)
	if len(colors_list) == 0:
		# print("return black")
		return(np.array([255,255,255]))


	weights = softmax(dist_list)
	# print("weights: ", weights)
	# print("colors: ", colors_list)
	# print("colors list: ", np.array(colors_list).shape)
	color_array =  np.array(colors_list)
	
	# Blend previous colors based on distance
	color_array = (color_array.T*weights).T

	return np.sum(color_array, axis=0)


def get_dict_panorama(index, frames_dict):

	return frames_dict[str(index)]



def main():

	create_consistency_image(0, 4, "../image_sets/images2_2_HD")

# print(target_im.shape)
# cv2.imshow("Source image: ", target_im)
# cv2.imshow("image 0: ", source_imgs[0][1])
# cv2.imshow("image 1: ", source_imgs[1][1])
# cv2.imshow("image 2: ", source_imgs[2][1])
# cv2.imshow("UV map: ", UV_map)
# cv2.waitKey(0)
if __name__ == '__main__':
	main()