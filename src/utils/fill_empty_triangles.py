import sys, os
import argparse
import math
import numpy as np

sys.path.insert(0, os.getcwd() + "/../../libigl/python/")

import pyigl as igl

# from shared import TUTORIAL_SHARED_PATH, check_dependencies
from iglhelpers import *
import math
import cv2

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

parser = argparse.ArgumentParser()
# parser.add_argument('--obj_path', type=str, default="../fake_sf.obj", help="Location of obj filee")
parser.add_argument('--obj_path', type=str, default="../scenes/scene1/senario_2_uv.obj", help="Location of obj file")
parser.add_argument('--image_path', type=str, default="pano.png", help="image to project on 3d object location")
parser.add_argument('--out_path', type=str, default="pano.png", help="image to project on 3d object location")
parser.add_argument('--uv_map_out_path', type=str, default="scene_UV_out.png", help="UV mapping")
parser.add_argument('--visibility_map_path', type=str, default="scene_UV_out.png", help="UV mapping")
pars = parser.parse_args()

# Load Mesh 
V = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
TC = igl.eigen.MatrixXd()
TT = igl.eigen.MatrixXi()
TTi = igl.eigen.MatrixXi()
FTC = igl.eigen.MatrixXi()
CN = igl.eigen.MatrixXd()
FN = igl.eigen.MatrixXi()
PFN = igl.eigen.MatrixXd()
#distance = igl.eigen.MatrixXd()

igl.readOBJ(pars.obj_path, V, TC, CN, F, FTC, FN)

#igl.all_pairs_distances(V,V,closest)

Vers = np.asarray(V)
Facs = np.asarray(F)
Vuvs = np.asarray(TC)
Vns = np.asarray(CN)
Fuvs_id	 = np.asarray(FTC)
Fns = np.asarray(FN)
# closest = np.asarray(closest)
print('Vers: ', Vers)
# print('Closest point\: ', closest)

# Load UV map
# uv_img = np.flip(np.flip(cv2.imread(pars.image_path),0),1)
uv_img = cv2.flip(cv2.flip(cv2.rotate(cv2.imread(pars.image_path), cv2.ROTATE_90_COUNTERCLOCKWISE),1),0)
dim = np.array([2048, 2048])
# Fill half textured triangles 
# visibility = np.load(pars.visibility_map_path)
# print('Visibility: ', visibility, visibility.shape)

# for i in range(visibility.shape[1]):
#     for j in range(visibility.shape[2]):
#         if not visibility[:,i,j].any():
            
#             print(i,j,uv_img[i,j])
#             uv_img[i,j] = [211,0,0]
for face_id1, (_, fac_uv) in enumerate(zip(Facs[:], Fuvs_id[:])):
    
    v1_uvs = Vuvs[int(fac_uv[0])]*dim
    v2_uvs = Vuvs[int(fac_uv[1])]*dim
    v3_uvs = Vuvs[int(fac_uv[2])]*dim
    
    min_x = int(round(min([v1_uvs[0], v2_uvs[0], v3_uvs[0]])))-1
    max_x = int(round(max([v1_uvs[0], v2_uvs[0], v3_uvs[0]])))+1
    min_y = int(round(min([v1_uvs[1], v2_uvs[1], v3_uvs[1]])))-1
    max_y = int(round(max([v1_uvs[1], v2_uvs[1], v3_uvs[1]])))+1
    
    #prev_color = [211,211,211]
    colors = []
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            if x<0:
                x=0
            if x>2047:
                x=2047
            if y<0:
                y=0
            if y>2047:
                y=2047
            if not (uv_img[x,y,0] == 211 and uv_img[x,y,1] == 211 and uv_img[x,y,2] == 211):
                colors.append(uv_img[x,y])
    
    if len(colors) != 0:
        colors = np.array(colors).astype(np.uint8)
        prev_color = np.average(colors, axis=0).astype(np.uint8)
    else:
        prev_color = [211,211,211]
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            if x<0:
                x=0
            if x>2047:
                x=2047
            if y<0:
                y=0
            if y>2047:
                y=2047
            #print(x,y)
            v, w, u = barycentric_coords([x + 0.5, y + 0.5], v1_uvs, v2_uvs, v3_uvs)
            #print('X,Y: ', x,y)
            if v>=0 and w>=0 and u>=0:
                colors = []
                if (uv_img[x,y,0] == 211 and uv_img[x,y,1] == 211 and uv_img[x,y,2] == 211):
                    if not y+1>max_y:
                        # color1 = uv_img[x,y+1]
                        colors.append(uv_img[x,y+1])
                        if not x+1>max_x:
                        #     #color5 = uv_img[x+1,y+1]
                             colors.append(uv_img[x+1,y+1])
                        if not x-1<min_x:
                        #     # color6 = uv_img[x-1,y+1]
                             colors.append(uv_img[x-1,y+1])
                    if not y-1<min_y:
                        # color2 = uv_img[x,y-1]
                        colors.append(uv_img[x,y-1])
                        if not x+1>max_x:
                        #     # color7 = uv_img[x+1,y-1]
                            colors.append(uv_img[x+1,y-1])
                        if not x-1<min_x:
                        #     # color8 = uv_img[x-1,y-1]
                            colors.append(uv_img[x-1,y-1])
                    if not x+1>max_x:
                        # color3 = uv_img[x+1,y]
                        colors.append(uv_img[x+1,y])
                    if not x-1<min_x:  
                        # color4 = uv_img[x-1,y]
                        colors.append(uv_img[x-1,y])
                    
                    
                    colors_final = []
                    for c in colors:
                    # for c in [color1, color2, color3, color4, color5, color6, color7, color8]:
                        if not (c[0] == 211 and c[1] == 211 and c[2] == 211):
                            colors_final.append(c)
                    
                    if len(colors_final) == 0:
                        uv_img[x,y] = prev_color
                        
                        continue
                    
                    
                    # print('Colors: ', colors)
                    colors = np.array(colors_final).astype(np.uint8)
                    #print('Colors: ', colors)
                    f_color = np.average(colors, axis=0).astype(np.uint8)
                    #print('F color: ', f_color)
                    #if not (f_color[0] == 211 and f_color[1] == 211 and f_color[2] == 211):
                    prev_color = f_color
                    
                    print(prev_color)
                    uv_img[x,y] = prev_color
                    
                    #print(prev_color)
#                 else:
                    
#                     print('Prev color: ', uv_img[x,y], prev_color)
#                 #else:
#                 #    prev_color=uv_img[x,y]
               
#     # break

uv_img = cv2.flip(cv2.flip(cv2.rotate(uv_img, cv2.ROTATE_90_CLOCKWISE), 0), 1)
cv2.imwrite(os.path.join(pars.out_path,'filled_UV.png'), uv_img)
                

# Fill empty triangles