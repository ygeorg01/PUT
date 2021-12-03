#!/usr/bin/env python
import sys, os

# Add the igl library to the modules search path
sys.path.insert(0, os.getcwd() + "/../libigl/python/")

import pyigl as igl
import argparse
import math
from tqdm import tqdm
import multiprocessing
import ctypes
import numpy as np
import cv2
import open3d as o3d
import create_dict as dict_


def load_camera_info_np(directory, filename):  # , start=pars.start, offset=pars.offset):

    camera_RT_matrices = []
    camera_locations = []
    # filename = '000057_RTm.txt'
    RTmatrix = np.loadtxt(os.path.join(directory, filename))

    # Camera rotation and translation matrix
    camera_RT_matrices.append(np.loadtxt(os.path.join(directory, filename)))

    # World coordin<3 location
    camera_locations.append(np.dot(-1 * RTmatrix[:, :-1].T, RTmatrix[:, -1]))
    print('Camera loc: ', np.dot(-1 * RTmatrix[:, :-1].T, RTmatrix[:, -1]))
    return camera_RT_matrices, camera_locations


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


def load_panoramas(directory, filename):  # , start=pars.start, offset=pars.offset):

    # filename = '00056.png'
    im_x, im_y = (512, 256)
    # print('Directory: ', directory)
    im = cv2.imread(directory + filename)
    # print('Image shape: ', im.shape)

    A_source = igl.eigen.MatrixXuc(im_x, im_y)
    R_source = igl.eigen.MatrixXuc(im_x, im_y)
    G_source = igl.eigen.MatrixXuc(im_x, im_y)
    B_source = igl.eigen.MatrixXuc(im_x, im_y)

    images_R_source = []
    images_G_source = []
    images_B_source = []
    images_A_source = []

    igl.png.readPNG(directory + filename, R_source, G_source, B_source, A_source)
    images_R_source.append(np.copy(np.array(R_source)))
    images_G_source.append(np.copy(np.array(G_source)))
    images_B_source.append(np.copy(np.array(B_source)))
    images_A_source.append(np.copy(np.array(A_source)))

    # return images_R_source, images_G_source, images_B_source, images_A_source
    return cv2.imread(directory + filename)


def length_line(p1, p2):
    return math.sqrt(((p1 ** 2) + (p2 ** 2)))


def Equirectangular(point_3d_coord, x_scale, y_scale):
    x_or = ((math.atan2(point_3d_coord[0], -point_3d_coord[2]) + math.pi) / (2 * math.pi))
    x = x_or * (x_scale - 1)
    y_or = ((math.atan2(point_3d_coord[1], length_line(point_3d_coord[2], point_3d_coord[0])) + (
                math.pi / 2)) / math.pi)
    y = y_or * (y_scale - 1)

    return x, y


def load_mesh():
    # Initialize arrays
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    TC = igl.eigen.MatrixXd()
    FTC = igl.eigen.MatrixXi()
    CN = igl.eigen.MatrixXd()
    FN = igl.eigen.MatrixXi()
    PFN = igl.eigen.MatrixXd()

    # Load mesh
    igl.readOBJ('../Scenes/11/mesh_huer_best.obj', V, TC, CN, F, FTC, FN)

    # Store information as numpy array
    Vers = np.asarray(V)
    Facs = np.asarray(F)
    Vuvs = np.asarray(TC)
    Vns = np.asarray(CN)
    Fuvs_id = np.asarray(FTC)
    Fns = np.asarray(FN)

    return Vers, Facs, Vuvs, Vns, Fuvs_id, Fns, V, F, TC, FTC


def l2b(point_3d):
    return np.array([[point_3d[0], -point_3d[2], point_3d[1], 1]])


def compute_3d_coords(vp1, vp2, vp3, w, v, u):
    pixel_3d = [vp1[0] * u + vp2[0] * v + vp3[0] * w, vp1[1] * u + vp2[1] * v + vp3[1] * w,
                vp1[2] * u + vp2[2] * v + vp3[2] * w, 1]
    return pixel_3d


def transform_RT_blender(camera_matrix, point_3d):
    # Transform to blender coords
    minus_Z = l2b([point_3d[0], point_3d[1], point_3d[2]])

    # Tranform 3d point to camera coordinats
    minus_Z_camera_coords = np.dot(camera_matrix, minus_Z.T)

    return minus_Z_camera_coords


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def create_mapping(Vers, Facs, Vuvs, Fuvs_id):
    coords = []
    feats = []

    # Load UV map
    uv_map = cv2.imread('../Scenes/11/UVs/reprojection.png')
    uv_map = cv2.rotate(uv_map, cv2.ROTATE_90_CLOCKWISE)

    count = 0
    UV_to_3D = np.zeros((2048, 2048, 3)) - 1
    uv_offset = 0
    for fac, fac_uv in zip(Facs, Fuvs_id):

        vp1 = Vers[fac[0]]
        vp2 = Vers[fac[1]]
        vp3 = Vers[fac[2]]

        v1_uvs = Vuvs[int(fac_uv[0])] * [2048, 2048]
        v2_uvs = Vuvs[int(fac_uv[1])] * [2048, 2048]
        v3_uvs = Vuvs[int(fac_uv[2])] * [2048, 2048]

        x_uvs = np.array([v1_uvs[0], v2_uvs[0], v3_uvs[0]])
        y_uvs = np.array([v1_uvs[1], v2_uvs[1], v3_uvs[1]])

        min_index_x = np.argmin(x_uvs)
        max_index_x = np.argmax(x_uvs)
        min_index_y = np.argmin(y_uvs)
        max_index_y = np.argmax(y_uvs)

        # x_uvs[min_index_x] -= 2
        # x_uvs[max_index_x] += 2
        # y_uvs[min_index_y] -= 2
        # y_uvs[max_index_y] += 2

        min_x = max(0, int(round(x_uvs[min_index_x])))
        max_x = min(2048, int(round(x_uvs[max_index_x])))
        min_y = max(0, int(round(y_uvs[min_index_y])))
        max_y = min(2048, int(round(y_uvs[max_index_y])))

        # # Writing over other triangles
        # space = 1
        #
        # if min_index_x == 0:
        #     v1_uvs[0] -= space
        # if min_index_x == 1:
        #     v2_uvs[0] -= space
        # if min_index_x == 2:
        #     v3_uvs[0] -= space
        #
        # if max_index_x == 0:
        #     v1_uvs[0] += space
        # if max_index_x == 1:
        #     v2_uvs[0] += space
        # if max_index_x == 2:
        #     v3_uvs[0] += space
        #
        # if min_index_y == 0:
        #     v1_uvs[1] -= space
        # if min_index_y == 1:
        #     v2_uvs[1] -= space
        # if min_index_y == 2:
        #     v3_uvs[1] -= space
        #
        # if max_index_y == 0:
        #     v1_uvs[1] += space
        # if max_index_y == 1:
        #     v2_uvs[1] += space
        # if max_index_y == 2:
        #     v3_uvs[1] += space

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):

                v, w, u = barycentric_coords([x + 0.5, y + 0.5], v1_uvs, v2_uvs, v3_uvs)

                if v >= 0 and w >= 0 and u >= 0:

                    point_3d_coord = compute_3d_coords(vp1, vp2, vp3, w, v, u)

                    print('3D points: ', point_3d_coord)
                    UV_to_3D[x, y, :] = point_3d_coord[:-1]
                    # print('x/y/3d_coord: ', x, y, point_3d_coord[:-1])
                    coords.append(point_3d_coord[:-1])
                    # feats.append([255, 255, 255])
                    feats.append([211, 211, 211])
                    # print('color: ', uv_map[x,y]/255)
                    count += 1

        # if count > 10000:
        #     break

        # count += 1
    print(UV_to_3D)
    np.save('1', UV_to_3D)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(feats)
    pcd.estimate_normals()
    o3d.io.write_point_cloud('pcd_11.pcd', pcd)
    print('number of points in pcd: ', count)


if __name__ == '__main__':
    scene_id = '11'

    # Load mesh
    Vers, Facs, Vuvs, Vns, Fuvs_id, Fns, V, F, TC, FTC = load_mesh()

    create_mapping(Vers, Facs, Vuvs, Fuvs_id)
