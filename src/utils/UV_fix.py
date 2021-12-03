import cv2
import numpy as np
from tqdm import tqdm

def check_near(UV_map, UV_dense_map, steps, x_axis, y_axis, color):
    # print("In check")
    for step in range(steps):
        if UV_map.shape[1] > (y_axis+step): 
            # print(UV_map[x_axis, y_axis+step])
            if (UV_map[x_axis, y_axis+step, 0:3] == 211).all():
                # print("color to change: ", UV_map[x_axis, y_axis+step], color)
                # if (color == [0,0,255]).all():
                # print("color to change1: ", UV_map[x_axis, y_axis+step], color)
                UV_dense_map[x_axis, y_axis+step] = color
            # else:
            #     break

    for step in range(steps):
        if (y_axis-step) > 0: 
            if (UV_map[x_axis, y_axis-step, 0:3] == 211).all():
                # print("color to change: ", UV_map[x_axis, y_axis-step], color)
                # if (color == [0,0,255]).all():
                # print("color to change2: ", UV_map[x_axis, y_axis+step], color)
                UV_dense_map[x_axis, y_axis-step] = color
            # else:
            #     break

    for step in range(steps):
		# print("step: ", step)
		# print("asdasdas: ", x_axis+step)
		# print("222222222: ", y_axis)
        if UV_map.shape[0] > (x_axis+step): 
			# print("x axis: ", x_axis+step)
            if (UV_map[x_axis+step, y_axis, 0:3] == 211).all():
                # print("color to change: ", UV_map[x_axis+step, y_axis], color)
                UV_dense_map[x_axis+step, y_axis] = color
            # else:
            #     break

    for step in range(steps):
        if (x_axis-step) > 0: 
            if (UV_map[x_axis-step, y_axis, 0:3] == 211).all():
                # print("color to change: ", UV_map[x_axis-step, y_axis], color)
                UV_dense_map[x_axis-step, y_axis] = color
            # else:
            #     break

def fill_mesh(path):

    # UV_map = cv2.imread('scene_UV_out.png')
    UV_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Create new blank UV texture
    UV_dense_map = np.zeros((UV_map.shape[0], UV_map.shape[1], UV_map.shape[2]))
    UV_dense_map[:,:,0:3] = 211
    UV_dense_map[:,:,3] = UV_map[0,0,3]


    for i in tqdm(range(UV_map.shape[0])):
        for j in range(UV_map.shape[1]):
            if (UV_map[i,j,0:3] != 211).all():
                UV_dense_map[i,j]=UV_map[i,j]
                check_near(UV_map, UV_dense_map, 2, i, j, UV_map[i,j])
            
		# if (UV_map[i,j] == [0,0,0]).all():
			# mask[i,j] = 0
			# print(1)

    cv2.imwrite(path, UV_dense_map)

