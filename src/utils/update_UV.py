import cv2
import argparse
import json
import numpy as np
import os

	
# parser = argparse.ArgumentParser()
# parser.add_argument('--UV_path', type=str, default = "UV_map_12.png", help="Location of obj filee")
# parser.add_argument('--panorama_path', type=str, default = "00015.png", help="Location of obj filee")
# parser.add_argument('--dictionary_path', type=str, default = "/home/vcg/Desktop/Urban\ Enviroment\ Understanding/Mesh_Texturing/code/frames_dict.json")
# parser.add_argument('--pano_number', type=int, default = 13) 
# parser.add_argument('--pano_width', type=int, default = 512)
# parser.add_argument('--pano_height', type=int, default = 256)
# parser.add_argument('--output_dir', type=str, default='./UVs')
# pars = parser.parse_args()


# Load Blend-Net Output
def load_panorama(panorama_path, pano_number, pano_size):

	# print('Path to panorama: ', os.path.join(panorama_path, str(pano_number).zfill(5)+'.png'))
	# if pano_number == 'black':
	# pano = cv2.imread(os.path.join(panorama_path, 'black.png'))
	# else:
	print('Path to output pano: ', panorama_path)
	# pano = cv2.imread(os.path.join(panorama_path, str(pano_number).zfill(5)+'.png'))
	pano = cv2.imread(panorama_path)

	# print('Pano shape: ', pano.shape)
	# pano = cv2.resize(pano, pano_size, interpolation = cv2.INTER_CUBIC)
	return pano
	
# Load current UV map.
def load_UV_map(UV_paths):

	UV = cv2.imread(UV_paths, cv2.IMREAD_UNCHANGED)

	return UV

def load_dictionary(dictionary_path):

	with open(dictionary_path, 'r') as f:
		return json.load(f)

def softmax(x):

	"""Compute softmax values for each sets of scores in x."""
	
	e_x = np.exp(x - np.max(x))

	return e_x / e_x.sum()

def update_uv(panorama_path, pano_number, dictionary_path, output_path, UV_map, pano_size, step, UV_colors, inter):

	# print('panorama path: ', panorama_path, pano_number, pano_size)
	panorama = load_panorama(panorama_path, pano_number, pano_size)

	UV = cv2.flip(UV_map, 0)
	panorama = cv2.flip(panorama,0)

	print('Disctionary ID: ' , int(((pano_number)/step)))
	frame_dict = load_dictionary(dictionary_path+'/'+str(int(round((pano_number)/step)))+'_dict.json')
	# print(frame_dict.keys())
	# frame_dict = frames_dict[str(int(pano_number/step))]

	frames_dict = None

	count = 0 
	# percentage = 0.80
	percentage = 0.7

	for key, values in frame_dict.items():
		# print('Key value: ', key, values)
		x_p, y_p = key.split(',')
		for v in values:			
			r,g,b, x_1, x_2, y_1, y_2 = billinear_inter(panorama, float(v[3]), float(v[2]))

			#if (r[0]==0 and g[0]==0 and b[0]==0):
			#	print('x1,x2,y1,y2: ', x_1,x_2, y_1, y_2)
			#	continue
			#UV[v[1], v[0],:3] = [r[0],g[0],b[0]]
			
			if str(v[1])+','+str(v[0]) not in UV_colors.keys():
				
				if 1 >= percentage:
					UV[v[1], v[0],:3] = [r[0],g[0],b[0]]
					UV[v[1], v[0],3] = 255


					# if (r[0]==0 and g[0]==0 and b[0]==0):
						# print('Color: ', r[0],g[0],b[0])
						# print('Vector: ', x_1, x_2, y_1, y_2)
						
				else:
					if 1 > 1-percentage:

						UV_colors[str(v[1])+','+str(v[0])] = []
						UV_colors[str(v[1])+','+str(v[0])].append([float(v[-1]), int(r), int(g), int(b)])
			else:
				UV_colors[str(v[1])+','+str(v[0])].append([float(v[-1]), int(r), int(g), int(b)])
				colors = []
				dist = []
				weights = []
				count = 0
				max_ = float(percentage)
				min_ = float(1-percentage)
				for cam_list in UV_colors[str(v[1])+','+str(v[0])]:
					colors.append([cam_list[1], cam_list[2], cam_list[3]])

					w = (float(cam_list[0])-min_)/(max_- min_)
					weights.append(w)

				# print('in else')
				color_array = np.array(colors)
				weights = np.array(weights)
				#print('Weights1: ', weights)
				# if weights[0]<0.1 and weights[1]<0.1:
				print('Weights: ', weights)

				color_array = (color_array.T*weights).T
				# print('Color array: ', color_array)
				color = np.sum(color_array, axis=0)

				UV[v[1], v[0],:3] = [color[0], color[1], color[2]]
				UV[v[1], v[0],3] = 255
				print('Coord: ', v[1], v[0])

			count+=1

	UV = cv2.flip(UV, 0)

	kernel = np.ones((5, 5), np.uint8)

	mask_UV_render = UV.copy()
	UV_render = UV.copy()
	print('UV render shape: ', UV_render.shape)
	mask_UV_render[mask_UV_render[:, :, 3] == 0, :] = [0, 0, 0, 0]
	mask_UV_render[mask_UV_render[:, :, 3] != 0, :] = [1, 1, 1, 1]
	# Using cv2.erode() method
	print('mask_UV_render 2', mask_UV_render.shape)
	mask_UV_render = cv2.erode(mask_UV_render, kernel, cv2.BORDER_REFLECT)
	print('mask_UV_render 3', mask_UV_render.shape, (mask_UV_render == 0).shape)
	mask_UV_render = np.nonzero((mask_UV_render == 0))
	# print('mask_UV_render 3', mask_UV_render)
	# print('mask_UV_render 3', mask_UV_render[0].shape)
	# UV_render[mask_UV_render == 0] = [0, 0, 0, 0]
	UV_render[mask_UV_render] = 0
	# UV_render[:, :, (mask_UV_render[:, :, 0] == 0)] = [0]
	# UV_render[:, :, (mask_UV_render[:, :, 0] == 0)] = [0]
	# UV_render[:, :, (mask_UV_render[:, :, 0] == 0)] = [0]
	# cv2.imwrite(output_path + "/UV_render_" + str(pano_number + step).zfill(5) + ".png", UV_render)

	cv2.imwrite(output_path+"/UV_"+str(pano_number+step).zfill(5)+".png", UV)


	return output_path+"/"+str(pano_number).zfill(5)+".png", UV

def invert_dist(dist, color_array):
	dist = np.array(dist)
	# print('dist: ', dist, dist.shape)
	distance = np.sum(dist)
	c=[0,0,0]
	for dist, color in zip(dist, color_array):
		c +=  dist*color
		
	return c/distance
		# print('dist, color: ', dist, color)



def billinear_inter(panorama, x, y):

	scale_x, scale_y,_ = panorama.shape
	
	x *= (scale_x-1)
	y *= (scale_y-1)

	# print('x:  ', x)
	# print('y: ', y)

	x_1 = min([int(np.floor(x+0.00001)), scale_x-1])
	x_2 = min([int(np.ceil(x+0.00001)), scale_x-1])

	if x_1 == scale_x-1 and x_2 == scale_x-1:
		x_2-=1

	y_1 = min([int(np.floor(y+0.00001)), scale_y-1])
	y_2 = min([int(np.ceil(y+0.00001)), scale_y-1])

	if y_1 == scale_y-1 and y_2 == scale_y-1:
		y_2-=1

	#print(x_1,y_1,x_2,y_2)
	x_inter = np.array([[x_2-x, x-x_1]])
	y_inter = np.array([[y_2-y],[y-y_1]])

	R_matrrix = [[panorama[x_1, y_1, 0], panorama[x_1, y_2, 0]],[panorama[x_2, y_1, 0], panorama[x_2, y_2, 0]]]
	G_matrrix = [[panorama[x_1, y_1, 1], panorama[x_1, y_2, 1]],[panorama[x_2, y_1, 1], panorama[x_2, y_2, 1]]]
	B_matrrix = [[panorama[x_1, y_1, 2], panorama[x_1, y_2, 2]],[panorama[x_2, y_1, 2], panorama[x_2, y_2, 2]]]

	R = np.dot(np.dot(x_inter,R_matrrix),y_inter)
	# print(R)
	G = np.dot(np.dot(x_inter,G_matrrix),y_inter)
	B = np.dot(np.dot(x_inter,B_matrrix),y_inter)
	# print('rbg: ', R,G,B)
	return R.round(), G.round(), B.round(), x_1,x_2, y_1, y_2


# if __name__ == "__main__":

	# update_uv(pars.panorama_path, pars.scene_number, pars.pano_number, pars.dictionary_path, pars.UV_path, pars.output_path, UV_map, (pars.pano_width, pars.pano_height))