#!/usr/bin/env python
import sys, os
# Add the igl library to the modules search path
# sys.path.insert(0, os.getcwd() + "/../../libigl/python/")

# import pyigl as igl
import argparse
import math
from tqdm import tqdm
import multiprocessing
import ctypes
import numpy as np

import cv2
import igl
# import create_dict as dict_


parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=str, default="/home/visual-computing-1/Desktop/projects/texturing/scenes/scene1/senario_2_uv.obj", help="Location of obj filee")
parser.add_argument('--start', type=int, default=0, help="UV mapping")
parser.add_argument('--offset', type=int, default=150, help="UV mapping")
parser.add_argument('--visual', type=bool, default=False, help="Display map")
parser.add_argument('--fill_mesh', type=bool, default=True)
parser.add_argument('--create_dict', type=bool, default=False)
parser.add_argument('--panoram_size', type=tuple, default= (1024 , 512))
parser.add_argument('--UV_size', type=tuple, default= (2048, 2048))
parser.add_argument('--scene_number', type=int, default= 4)
parser.add_argument('--distance_threshold', type=int, default= 25)
parser.add_argument('--processes_number', type=int, default= 10)
parser.add_argument('--UV_output_path', type=str, default= 10)
parser.add_argument('--step', type=int, default= 1)
parser.add_argument('--no_dis_filter', type=bool, default = False)
parser.add_argument('--mesh_name', type=str, default = 'mesh')
pars = parser.parse_args()

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

def load_mesh():

    
    # Load mesh
    # igl.readOBJ(pars.scene_path+'/mesh.obj', V, TC, CN, F, FTC, FN)
    # obj = igl.read_obj(pars.scene_path+'/'+pars.mesh_name+'.obj')
    v, tc, n, f, ftc, fn = igl.read_obj(pars.scene_path+'/'+pars.mesh_name+'.obj')

    print('vertex: ', v.shape)
    a = igl.adjacency_matrix(f)
    print('A shape: ', a.shape, a[0])
    c = igl.connected_components(a)

    print('Connected components: ', len(c))
	# Inputs:
	# 	F  #F by simplex_size list of mesh faces (must be triangles)
	# Outputs:
	#  	TT   #F by #3 adjacent matrix, the element i,j is the id of the triangle
	#       adjacent to the j edge of triangle i
	#  	TTi  #F by #3 adjacent matrix, the element i,j is the id of edge of the
	#       triangle TT(i,j) that is adjacent with triangle i
    tt, tti = igl.triangle_triangle_adjacency(f)

    # print('TT shape: ', tt.shape)
    # c, counts = igl.facet_components(tt)

    return v, tc, n, f, ftc, fn, tt, tti

# def point2area(Vers, Facs):
def distance(c1,c2):

	return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def neighborhoods3D(v,f):

	face2neighborhoods = np.zeros((f.shape[0],1), dtype=int)
	seen = np.zeros((f.shape[0]), dtype=bool)
	n_id = 0
	neighborhood = []
	for face_id, face in enumerate(f):
		print(face_id)
		# print(face, face_id, v[face[0]], v[face[1]], v[face[2]])
		
		ver_coords = v[face[0]], v[face[1]], v[face[2]]
		center = (ver_coords[0] + ver_coords[1] + ver_coords[2])/3
		if seen[face_id]:
			continue

		seen[face_id] = True

		face2neighborhoods[face_id] = n_id
		neighborhood.append([])
		neighborhood[n_id].append(face_id)
		for f_id, face_2 in enumerate(f):
			# print('F_idd: ', f_id)
			if seen[f_id]:
				continue


			ver_coords2 = v[face_2[0]], v[face_2[1]], v[face_2[2]]
			center2 = (ver_coords2[0] + ver_coords2[1] + ver_coords2[2])/3

			if distance(center, center2) < 40:
				face2neighborhoods[f_id] = n_id
				seen[f_id] = True
				neighborhood[n_id].append(f_id)

		n_id+=1

		if face_id>100:
			break

	for l in neighborhood:
		print('List, count: ', l)


def connected_components(f, tt):

    seen = np.zeros((f.shape[0]), dtype=bool)

    connected = []
    face2component = np.zeros((f.shape[0],1)) - 1
    component_id = 0
    component_count = []
    component_list = []
    for face_id, face in enumerate(f):

        if seen[face_id]:
            continue

        connected.append(face_id)
        count=0
        component_count.append(0)
        component_list.append([])
        while len(connected) != 0:

            f_id = connected[0]
    		
            connected.pop()
            if not seen[f_id]:
    			# continue
                component_list[component_id].append(f_id)
                seen[f_id] = True
                count+=1
                component_count[component_id]+=1
                face2component[f_id] = component_id
                
                for n_f_id in tt[f_id]:
                    if not seen[n_f_id]:
                        connected.append(n_f_id)

        component_id += 1

        if face_id > 1000:
        	break

    print('Compopnent count: ', component_count)
    for l,c in zip(component_list, component_count):
        if c>5:
            print('List, count: ', l, c)

if __name__ == '__main__':

	# Load Mesh
    v, tc, n, f, ftc, fn, tt, tti = load_mesh()
    # obj = load_mesh()

    # Load vizibility matrix
    # visibility_matrix = np.load(os.path.join(pars.scene_path, "visibility.npy"))

    # neighborhoods3D(f, tt)

    # # print('Compopnent count: ', component_count)
    # for l,c in zip(component_list, component_count):
    # 	if c>10:
    # 		print('List, count: ', l, c)

    UV = cv2.imread(pars.scene_path+'/UVs/filled_UV.png')
    new_UV = np.zeros((UV.shape))
    # # print(UV.shape)

    # face_ids = [597, 677, 602, 676, 598, 699, 599, 685, 665, 603, 674, 655, 675, 600]
    face_ids = [76, 80, 82, 199, 1520, 4938, 4943, 4948, 4957, 4965, 4966, 4968, 5043, 5045, 5049, 5059, 6616, 6618, 29497, 29498, 29499, 36305, 36341, 36343, 36344, 41012, 42706, 42790, 204679, 204742, 346879, 346948, 346950, 350068, 350164, 350200, 350210, 350252, 350512, 350640, 350644, 350651, 350657, 350660, 350678, 350680, 350682, 350683, 350684, 350696, 350697, 350699, 350732, 350766, 350799, 351244, 351421, 351451, 351505, 353185, 353186, 353187, 353189, 354681, 354748, 354833, 354834, 354836, 354838, 354840, 355476, 358720, 358744, 358775, 359332, 359335]

    # # face_ids = [956, 1266, 957, 1291, 958, 1265]
    count = 0 
    for f_id in face_ids:
        print(f_id)
        tc_ids = ftc[f_id]
        uv_coords = [tc[tc_ids[0]]*[2048, 2048], tc[tc_ids[1]]*[2048, 2048], tc[tc_ids[2]]*[2048, 2048]]
        # print(f[f_id], uv_coords)

        x_uvs = np.array([uv_coords[0][0],uv_coords[1][0],uv_coords[2][0]])
        y_uvs = np.array([uv_coords[0][1],uv_coords[1][1],uv_coords[2][1]])

        min_x = int(round(np.min(x_uvs)))
        max_x = int(round(np.max(x_uvs)))
        min_y = int(round(np.min(y_uvs)))
        max_y = int(round(np.max(y_uvs)))

        for x in range(min_x,max_x):
            for y in range(min_y, max_y):
                print(x,y)
                v, w, u = barycentric_coords([x + 0.5, y + 0.5], uv_coords[0], uv_coords[1], uv_coords[2])
                
                if v>=0 and w>=0 and u>=0:
                    print('Count: ', count)
                    print('Color: ', UV[x,y])
                    new_UV[x,y] = UV[x,y]
                    count+=1
    
    cv2.imwrite('/home/vcg/Desktop/component.png', new_UV)
    # Load UV 
    # UV = cv2.imread(os.path.join(pars.scene_path, 'UVs', 'reprojection.png'))



    # print('UV shape: ', UV.shape)
    # print('Visibility matrix shape: ', visibility_matrix.shape)