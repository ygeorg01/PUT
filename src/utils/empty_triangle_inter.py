import sys, os
import argparse
import math
import numpy as np

# Add the igl library to the modules search path
sys.path.insert(0, os.getcwd() + "/../../libigl/python/")

import pyigl as igl

from shared import TUTORIAL_SHARED_PATH, check_dependencies
from iglhelpers import *
import math
# from reprojection_eigen import load_camera_info_np


def load_camera_info_np(directory):

	camera_RT_matrices = []
	camera_locations = []
	count = 0
	for filename in sorted(os.listdir(directory)):
			# if count == 5 or count == 98 or count == 140:
				# print("filename: ", filename)
			RTmatrix = np.loadtxt(os.path.join(directory,filename))
				# Camera rotation and translation matrix
			camera_RT_matrices.append(np.loadtxt(os.path.join(directory,filename)))
				# World coordin<3 location
			camera_locations.append(np.dot(-1*RTmatrix[:,:-1].T, RTmatrix[:,-1]))
			count+=1
			# if count>140:
				# break
	return camera_RT_matrices, camera_locations


def get_colors(active_faces, R, G, B, Facs, Fuvs_id, Vuvs, shape):
	# print("In get colors")
	print("Facs: ", Facs.shape)
	faces_colors = []
	for face_id1, (_, fac_uv) in enumerate(zip(Facs[:], Fuvs_id[:])):
		# print("fece id: ", face_id1)
		# print("facs uv: ", fac_uv)
		if active_faces[face_id1] == 1:
			# Scale coordinates uv coordinates
			v1_uvs = Vuvs[int(fac_uv[0])]*shape
			r1 =  R[int(v1_uvs[0]), int(v1_uvs[1])]
			g1 =  G[int(v1_uvs[0]), int(v1_uvs[1])]
			b1 =  B[int(v1_uvs[0]), int(v1_uvs[1])]

			v2_uvs = Vuvs[int(fac_uv[1])]*shape
			r2 =  R[int(v2_uvs[0]), int(v2_uvs[1])]
			g2 =  G[int(v2_uvs[0]), int(v2_uvs[1])]
			b2 =  B[int(v2_uvs[0]), int(v2_uvs[1])]


			v3_uvs = Vuvs[int(fac_uv[2])]*shape
			r3 =  R[int(v3_uvs[0]), int(v3_uvs[1])]
			g3 =  G[int(v3_uvs[0]), int(v3_uvs[1])]
			b3 =  B[int(v3_uvs[0]), int(v3_uvs[1])]

			center_tri = [(v1_uvs[0] + v2_uvs[0] + v3_uvs[0]) / 3, (v1_uvs[1] + v2_uvs[1] + v3_uvs[1]) / 3]
			r_c =  R[int(center_tri[0]), int(center_tri[1])]
			g_c =  G[int(center_tri[0]), int(center_tri[1])]
			b_c =  B[int(center_tri[0]), int(center_tri[1])]

			# print("Points: ", v1_uvs, v2_uvs, v3_uvs, center_tri)
			# print("Colors: ", R[int(v1_uvs[0]), int(v1_uvs[1])])

			faces_colors.append([(r1, g1, b1), (r2, g2, b2), (r3, g3, b3), (r_c, g_c, b_c)])
		else:
			faces_colors.append([(0,0,0)])

	return faces_colors


def softmax(x):

	"""Compute softmax values for each sets of scores in x."""
	
	e_x = np.exp(x - np.max(x))

	return e_x / e_x.sum()

def interpolate(color_edge_0, color_edge_1, color_edge_2, v, u, w):

	color = 0
	# print("colors: ", color_edge_0, color_edge_1, color_edge_2)
	# print("colors len: ", len(color_edge_0), len(color_edge_1), len(color_edge_2))

	# print("v,u,w: %f, %f, %f" % (v, u,w))
	

	if (len(color_edge_0) != 0) and (len(color_edge_1) != 0) and (len(color_edge_2) != 0):
		weights = softmax(np.array([v,u,w]))
		# print("v u w: ", softmax(np.array([v,u,w])))
		color = (color_edge_0[0] * weights[0] + color_edge_1[0] * weights[1] + color_edge_2[0] * weights[2],\
				color_edge_0[1] * weights[0] + color_edge_1[1] * weights[1] + color_edge_2[1] * weights[2],\
				color_edge_0[2] * weights[0] + color_edge_1[2] * weights[1] + color_edge_2[2] * weights[2])

	elif (len(color_edge_0) != 0) and (len(color_edge_1) != 0) and (len(color_edge_2) == 0):
		weights = softmax(np.array([v,u]))
		# print("v u: ", softmax(np.array([v,u])))
		color = (color_edge_0[0] * weights[0] + color_edge_1[0] * weights[1],\
				color_edge_0[1] * weights[0] + color_edge_1[1] * weights[1],\
				color_edge_0[2] * weights[0] + color_edge_1[2] * weights[1],)

	elif (len(color_edge_0) == 0) and (len(color_edge_1) != 0) and (len(color_edge_2) != 0):
		weights = softmax(np.array([u,w]))
		# print("u w: ", softmax(np.array([u,w])))
		color = (color_edge_1[0] * weights[0] + color_edge_2[0] * weights[1],\
				color_edge_1[1] * weights[0] + color_edge_2[1] * weights[1],\
				color_edge_1[2] * weights[0] + color_edge_2[2] * weights[1])

	elif (len(color_edge_0) != 0) and (len(color_edge_1) == 0) and (len(color_edge_2) != 0):
		weights = softmax(np.array([v,w]))
		# print("v w: ", softmax(np.array([v,w])))
		color = (color_edge_0[0] * weights[0] + color_edge_2[0] * weights[1],\
				color_edge_0[1] * weights[0] + color_edge_2[1] * weights[1],\
				color_edge_0[2] * weights[0] + color_edge_2[2] * weights[1])
	elif (len(color_edge_0) != 0):
		# weights = softmax(np.array([v]))
		color = (color_edge_0[0], color_edge_0[1],  color_edge_0[2])
	elif (len(color_edge_1) != 0):
	# 	# weights = softmax(np.array([v,u,w]))
		color = (color_edge_1[0], color_edge_1[1], color_edge_1[2])
	elif (len(color_edge_2) != 0):
	# 	# weights = softmax(np.array([v,u,w]))
		color = (color_edge_2[0], color_edge_2[1], color_edge_2[2])

	# print("Final color: ", color)
	return color
def distance(p1,p2):
	
	return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def visibility_filtering_normals(camera_locations, V, F, Vers, Facs, distance_threshold):

	visibility_matrix = np.zeros((len(camera_locations), Facs.shape[0]))

	camera_positions = []
	ray_pos = []
	ray_dir = []
	cam_ids = []
	target_pos = []
	target_cam_index = []
	active_faces = np.zeros((Facs.shape[0]))

	for cam_id, cam_l in enumerate(camera_locations):
		camera_positions.append(np.array([cam_l[0], cam_l[2], -cam_l[1]]))
		ray_pos.append([])
		ray_dir.append([])
		target_pos.append([])
		target_cam_index.append([])
		cam_ids.append(cam_id)
	
	for fac_id, fac in enumerate(Facs):
		vp1 = Vers[fac[0]]
		vp2 = Vers[fac[1]]
		vp3 = Vers[fac[2]]
		centroid = np.array([(vp1[0]+vp2[0]+vp3[0])/3, (vp1[1]+vp2[1]+vp3[1])/3, (vp1[2]+vp2[2]+vp3[2])/3])

		for pos, direction, cam_id in zip(camera_positions, (centroid - camera_positions), cam_ids):
			if distance(camera_positions[cam_id], centroid) <= distance_threshold:
				ray_pos[cam_id].append(pos)
				ray_dir[cam_id].append(direction)
				target_pos[cam_id].append(fac)
				target_cam_index[cam_id].append(cam_id)

	import itertools
	ray_dir_combine = itertools.chain.from_iterable(ray_dir)
	ray_pos_combine = itertools.chain.from_iterable(ray_pos)
	targets_pos = itertools.chain.from_iterable(target_pos)
	targets_cam_index = itertools.chain.from_iterable(target_cam_index)

	hits = igl.embree.line_mesh_intersection(igl.eigen.MatrixXd(list(ray_pos_combine)), igl.eigen.MatrixXd(list(ray_dir_combine)), V, F)

	for target, hit, cam_id in zip(targets_pos,np.asarray(hits), targets_cam_index):
		visibility_matrix[cam_id, int(hit[0])] = 1
		active_faces[int(hit[0])] = 1

	np.save("active_faces", active_faces)
	np.save("visibility_matrix", visibility_matrix)
	return visibility_matrix, active_faces

def barycentric_coords(point, a, b, c):
	
	v1 = b - a
	v2 = c - a
	v3 = point - a

	d00 = np.dot(v1, v1)
	d01 = np.dot(v1, v2)
	d11 = np.dot(v2, v2)
	d20 = np.dot(v3, v1)
	d21 = np.dot(v3, v2)
	denom = d00 * d11 - d01 * d01
	v = (d11 * d20 - d01 * d21) / denom
	w = (d00 * d21 - d01 * d20) / denom
	u = 1 - v - w

	return v, w, u

# def cast_color(color_edge_0, color_edge_1, color_edge_2, v, w, v):

parser = argparse.ArgumentParser()
# parser.add_argument('--obj_path', type=str, default="../fake_sf.obj", help="Location of obj filee")
parser.add_argument('--obj_path', type=str, default="../scenes/scene1/senario_2_uv.obj", help="Location of obj file")
parser.add_argument('--image_path', type=str, default="pano.png", help="image to project on 3d object location")
parser.add_argument('--uv_map_out_path', type=str, default="scene_UV_out.png", help="UV mapping")
pars = parser.parse_args()

camera_RT_matrices, camera_locations = load_camera_info_np("../scenes/scene1/camera_RT_2")
# print('Camera RT: ', camera_RT_matrices)
# Initialize arrays
V = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
TC = igl.eigen.MatrixXd()
TT = igl.eigen.MatrixXi()
TTi = igl.eigen.MatrixXi()
FTC = igl.eigen.MatrixXi()
CN = igl.eigen.MatrixXd()
FN = igl.eigen.MatrixXi()
PFN = igl.eigen.MatrixXd()

# Store information as numpy array



# # Load mesh
igl.readOBJ(pars.obj_path, V, TC, CN, F, FTC, FN)
Vers = np.asarray(V)
Facs = np.asarray(F)
Vuvs = np.asarray(TC)
Vns = np.asarray(CN)
Fuvs_id	 = np.asarray(FTC)
Fns = np.asarray(FN)
print('Mesh loaded..')
# TO DO: get neighboring faces
# # neighboors_martix = parse_neigboors_info("neighboors.txt")
igl.triangle_triangle_adjacency(F,TT,TTi)
TT = np.array(TT)
TTi = np.array(TTi)

# Faces = np.asarray(F)
# print("Faces: ", Faces)
# file = open("scene_neighboors.txt", "a")

x_dim = 12288
y_dim = 6144
dim = np.array([x_dim,y_dim])

R_source_N = igl.eigen.MatrixXuc(x_dim, y_dim)*0
G_source_N = igl.eigen.MatrixXuc(x_dim, y_dim)*0
B_source_N = igl.eigen.MatrixXuc(x_dim, y_dim)*0
A_source_N = igl.eigen.MatrixXuc(x_dim, y_dim)*0

R_source_O = igl.eigen.MatrixXuc(x_dim, y_dim)
G_source_O = igl.eigen.MatrixXuc(x_dim, y_dim)
B_source_O = igl.eigen.MatrixXuc(x_dim, y_dim)
A_source_O = igl.eigen.MatrixXuc(x_dim, y_dim)
# Load Texture Map O
igl.png.readPNG("scene_UV_inter1.png", R_source_O, G_source_O, B_source_O, A_source_O)

threshold = 20
# visibility_matrix, active_faces = visibility_filtering_normals(camera_locations, V, F, Vers, Facs, threshold)
visibility_matrix = np.load('visibility_matrix.npy') 
active_faces = np.load('active_faces.npy')
print("Visibility matrix ready")
print("Fuvs_id: ", Fuvs_id)

# Get colors from textured faces(vertex and center colors)
faces_color_matrix = get_colors(active_faces, R_source_O, G_source_O, B_source_O, Facs, Fuvs_id, Vuvs, dim)


# For each face find neighboors
for face_id1, (_, fac_uv) in enumerate(zip(Facs[:], Fuvs_id[:])):
	# Active faces
	color_edge_1 = []
	color_edge_2 = []
	color_edge_0 = []

	print("Face Id: ", face_id1)
	if active_faces[face_id1] == 0:
		# Find neighboors
		
		if TT[face_id1][0] != -1:
			if face_id1 in TT[TT[face_id1][0]]:
				face_id2 = TT[face_id1][0]
				# print("index of edge : ", TT[face_id2].tolist().index(face_id1))
				index_color = TT[face_id2].tolist().index(face_id1)
				# print("colors: ", faces_color_matrix[face_id2])
				if len(faces_color_matrix[face_id2])>1:
					if index_color == 0:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][0])
						color = [(faces_color_matrix[face_id2][0][0], faces_color_matrix[face_id2][0][1], faces_color_matrix[face_id2][0][2])\
							,(faces_color_matrix[face_id2][1][0], faces_color_matrix[face_id2][1][1],faces_color_matrix[face_id2][1][2])]
					if index_color == 1:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][1])
						color = [(faces_color_matrix[face_id2][1][0], faces_color_matrix[face_id2][1][1], faces_color_matrix[face_id2][1][2])\
							,(faces_color_matrix[face_id2][2][0], faces_color_matrix[face_id2][2][1],faces_color_matrix[face_id2][2][2])]
					if index_color == 2:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][2])
						color = [(faces_color_matrix[face_id2][0][0], faces_color_matrix[face_id2][0][1], faces_color_matrix[face_id2][0][2])\
							,(faces_color_matrix[face_id2][2][0], faces_color_matrix[face_id2][2][1],faces_color_matrix[face_id2][2][2])]
					# print("color: ", color)
					# print("color: ",)
					if TTi[face_id1][0] == 0 :
						color_edge_0.append(color[0])
						color_edge_1.append(color[1])
					if TTi[face_id1][0] == 1:
						color_edge_1.append(color[0])
						color_edge_2.append(color[1])
					else:
						color_edge_0.append(color[0])
						color_edge_2.append(color[1])

		if TT[face_id1][1] != -1:
			if face_id1 in TT[TT[face_id1][1]]:
				face_id2 = TT[face_id1][1]
				# print("index of edge : ", TT[face_id2].tolist().index(face_id1))
				index_color = TT[face_id2].tolist().index(face_id1)
				# print("colors: ", faces_color_matrix[face_id2])
				if len(faces_color_matrix[face_id2])>1:
					if index_color == 0:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][0])
						color = [(faces_color_matrix[face_id2][0][0], faces_color_matrix[face_id2][0][1], faces_color_matrix[face_id2][0][2])\
							,(faces_color_matrix[face_id2][1][0], faces_color_matrix[face_id2][1][1],faces_color_matrix[face_id2][1][2])]
					if index_color == 1:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][1])
						color = [(faces_color_matrix[face_id2][1][0], faces_color_matrix[face_id2][1][1], faces_color_matrix[face_id2][1][2])\
							,(faces_color_matrix[face_id2][2][0], faces_color_matrix[face_id2][2][1],faces_color_matrix[face_id2][2][2])]
					if index_color == 2:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][2])
						color = [(faces_color_matrix[face_id2][0][0], faces_color_matrix[face_id2][0][1], faces_color_matrix[face_id2][0][2])\
							,(faces_color_matrix[face_id2][2][0], faces_color_matrix[face_id2][2][1],faces_color_matrix[face_id2][2][2])]
					# print("color: ", color)
					# print("color: ",)
					if TTi[face_id1][1] == 0 :
						color_edge_0.append(color[0])
						color_edge_1.append(color[1])
					if TTi[face_id1][1] == 1:
						color_edge_1.append(color[0])
						color_edge_2.append(color[1])
					else:
						color_edge_0.append(color[0])
						color_edge_2.append(color[1])

		if TT[face_id1][2] != -1:
			# print("Neighboors color 2: ", TT[TT[face_id1][2]])
			# print(" Side of neighboring edge 2: ", TTi[TT[face_id1][2]])
			if face_id1 in TT[TT[face_id1][2]]:
				face_id2 = TT[face_id1][2]
				# print("index of edge : ", TT[face_id2].tolist().index(face_id1))
				index_color = TT[face_id2].tolist().index(face_id1)
				# print("colors: ", faces_color_matrix[face_id2])
				if len(faces_color_matrix[face_id2])>1:
					if index_color == 0:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][0])
						color = [(faces_color_matrix[face_id2][0][0], faces_color_matrix[face_id2][0][1], faces_color_matrix[face_id2][0][2])\
							,(faces_color_matrix[face_id2][1][0], faces_color_matrix[face_id2][1][1],faces_color_matrix[face_id2][1][2])]
					if index_color == 1:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][1])
						color = [(faces_color_matrix[face_id2][1][0], faces_color_matrix[face_id2][1][1], faces_color_matrix[face_id2][1][2])\
							,(faces_color_matrix[face_id2][2][0], faces_color_matrix[face_id2][2][1],faces_color_matrix[face_id2][2][2])]
					if index_color == 2:
						# print(faces_color_matrix[face_id2][0], faces_color_matrix[face_id2][2])
						color = [(faces_color_matrix[face_id2][0][0], faces_color_matrix[face_id2][0][1], faces_color_matrix[face_id2][0][2])\
							,(faces_color_matrix[face_id2][2][0], faces_color_matrix[face_id2][2][1],faces_color_matrix[face_id2][2][2])]
					# print("color: ", color)
					# print("color: ",)
					if TTi[face_id1][2] == 0 :
						color_edge_0.append(color[0])
						color_edge_1.append(color[1])
					if TTi[face_id1][2] == 1:
						color_edge_1.append(color[0])
						color_edge_2.append(color[1])
					else:
						color_edge_0.append(color[0])
						color_edge_2.append(color[1])

		if len(color_edge_0) != 0 or len(color_edge_1) != 0 or len(color_edge_2) != 0:
			# print("Colors in the edges: ", color_edge_0, color_edge_1, color_edge_2)
			
			# print("Color mean for tuples: ", [sum(t) / len(t) for t in zip(*color_edge_0)], [sum(t) / len(t) for t in zip(*color_edge_1)], [sum(t) / len(t) for t in zip(*color_edge_2)])
			
			color_edge_0 = [sum(t) / len(t) for t in zip(*color_edge_0)]
			color_edge_1 = [sum(t) / len(t) for t in zip(*color_edge_1)]
			color_edge_2 = [sum(t) / len(t) for t in zip(*color_edge_2)]

			v1_uvs = Vuvs[int(fac_uv[0])]*dim
			v2_uvs = Vuvs[int(fac_uv[1])]*dim
			v3_uvs = Vuvs[int(fac_uv[2])]*dim

			min_x = int(min([v1_uvs[0], v2_uvs[0], v3_uvs[0]]))
			max_x = int(max([v1_uvs[0], v2_uvs[0], v3_uvs[0]]))
			min_y = int(min([v1_uvs[1], v2_uvs[1], v3_uvs[1]]))
			max_y = int(max([v1_uvs[1], v2_uvs[1], v3_uvs[1]]))

			for x in range(min_x,max_x):
				for y in range(min_y, max_y):
					# print("x,y: ", x,y)
					v, w, u = barycentric_coords([x + 0.5, y + 0.5], v1_uvs, v2_uvs, v3_uvs)
					if v>=0 and w>=0 and u>=0:
						color = interpolate(color_edge_0, color_edge_1, color_edge_2, v, w, u)
						# print(int(round(color[0])), int(round(color[1])), int(round(color[2])))
						if R_source_O[x,y] == G_source_O[x,y] and G_source_O[x,y] == B_source_O[x,y]:
							R_source_O[x,y] = int(round(color[0]))
							G_source_O[x,y] = int(round(color[1]))
							B_source_O[x,y] = int(round(color[2]))

igl.png.writePNG(R_source_O, G_source_O, B_source_O, A_source_O, "scene_UV_inter.png")

		# color_edge_3 = 

        # Get neighboors color

		# Interpolate colors to empty triangle

		# Change colors on texture Map N