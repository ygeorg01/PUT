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
import create_dict as dict_


def load_camera_info_np(directory, filename):  # , start=pars.start, offset=pars.offset):

    # camera_RT_matrices = []
    # camera_locations = []
    # count = 0
    # for filename in sorted(os.listdir(directory)):
    #     if count<start+offset and count>=start and (count%pars.step == 0):
    #        RTmatrix = np.loadtxt(os.path.join(directory,filename))
    #        print('filenames: ', filename)
    # # Camera rotation and translation matrix
    #        camera_RT_matrices.append(np.loadtxt(os.path.join(directory,filename)))
    # # World coordin<3 location
    #        camera_locations.append(np.dot(-1*RTmatrix[:,:-1].T, RTmatrix[:,-1]))
    #     count+=1

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
    print('Directory: ', directory + filename)
    return cv2.imread(directory + filename)


def length_line(p1, p2):
    return math.sqrt(((p1 ** 2) + (p2 ** 2)))


def Equirectangular(point_3d_coord, x_scale, y_scale):
    x_or = ((math.atan2(point_3d_coord[0], -point_3d_coord[2]) + math.pi) / (2 * math.pi))
    x = x_or * (x_scale - 1)
    y_or = ((math.atan2(point_3d_coord[1], length_line(point_3d_coord[2], point_3d_coord[0])) + (
                math.pi / 2)) / math.pi)
    y = y_or * (y_scale - 1)

    return x, y, x_or,  y_or


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
    # igl.readOBJ(pars.scene_path+'/mesh.obj', V, TC, CN, F, FTC, FN)
    print('Load obj: ', '../scenes/' + str(scene_id) + '/' + obj_name + '.obj', V, TC, CN, F, FTC, FN)
    igl.readOBJ('../scenes/' + str(scene_id) + '/' + obj_name + '.obj', V, TC, CN, F, FTC, FN)
    # igl.readOBJ('../Scenes/scene4_clean/mesh.obj', V, TC, CN, F, FTC, FN)

    # Store information as numpy array
    Vers = np.asarray(V)
    Facs = np.asarray(F)
    Vuvs = np.asarray(TC)
    Vns = np.asarray(CN)
    Fuvs_id = np.asarray(FTC)
    Fns = np.asarray(FN)

    # Compute NOrmals
    # igl.per_face_normals(V,F,PFN)
    # face_normals =  np.asarray(PFN)

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


def look_up_matrix(camera_locations, V, F, Vers, Facs, Vuvs, Fuvs_id, RT_matrix, filename):
    camera_positions = []
    ray_pos = []
    ray_dir = []
    cam_ids = []
    target_pos = []
    target_cam_index = []
    y_index = []
    x_index = []
    face_index_array = []
    x_pixel = []
    y_pixel = []
    y_or_l = []
    x_or_l = []

    for cam_id, cam_l in enumerate(camera_locations):
        camera_positions.append(np.array([cam_l[0], cam_l[2], -cam_l[1]]))
        ray_pos.append([])
        ray_dir.append([])
        target_pos.append([])
        target_cam_index.append([])
        cam_ids.append(cam_id)
        x_index.append([])
        y_index.append([])
        face_index_array.append([])
        x_pixel.append([])
        y_pixel.append([])
        y_or_l.append([])
        x_or_l.append([])

    face_index = 0
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

        x_uvs[min_index_x] -= 2
        x_uvs[max_index_x] += 2
        y_uvs[min_index_y] -= 2
        y_uvs[max_index_y] += 2

        min_x = max(0, int(x_uvs[min_index_x]))
        max_x = min(2048, int(x_uvs[max_index_x]))
        min_y = max(0, int(y_uvs[min_index_y]))
        max_y = min(2048, int(y_uvs[max_index_y]))

        # counter_xy = 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):

                v, w, u = barycentric_coords([x + 0.5, y + 0.5], v1_uvs, v2_uvs, v3_uvs)

                if v >= 0 and w >= 0 and u >= 0:
                    point_3d_coord = compute_3d_coords(vp1, vp2, vp3, w, v, u)

                    point_3d_cam_coord = transform_RT_blender(RT_matrix, point_3d_coord)

                    x_im_coord, y_im_coord, x_or, y_or = Equirectangular(point_3d_cam_coord, 512, 256)

                    # print('3D point: ', np.array(point_3d_coord[:3]))
                    # print('Camera position: ', camera_positions)

                    # for pos, direction, cam_id in zip(camera_positions, (np.array(point_3d_coord[:3]) - camera_positions), cam_ids):

                    pos = camera_positions[0]
                    direction = (np.array(point_3d_coord[:3]) - camera_positions[0])
                    # print('Direction: ', direction)
                    # print('Point 3d coords: ', point_3d_coord[:3])
                    # print('camera : ', camera_positions[0])

                    # print(direction)
                    cam_id = cam_ids[0]

                    # print('X coord: ', x_im_coord)

                    if distance(camera_positions[cam_id], np.array(point_3d_coord[:3])) <= 40:
                        # print('Distance: ', distance(camera_positions[cam_id], np.array(point_3d_coord[:3])))
                        ray_pos[cam_id].append(pos)
                        ray_dir[cam_id].append(direction)
                        target_pos[cam_id].append(fac)
                        target_cam_index[cam_id].append(cam_id)
                        y_index[cam_id].append(y)
                        x_index[cam_id].append(x)
                        y_or_l[cam_id].append(y_or)
                        x_or_l[cam_id].append(x_or)
                        face_index_array[cam_id].append(face_index)
                        x_pixel[cam_id].append(int(round(x_im_coord)))
                        y_pixel[cam_id].append(int(round(y_im_coord)))

        face_index += 1

    look_up_m = np.zeros((2048, 2048, 2)) - 1
    pano_UV = np.zeros((512, 256, 2)) - 1

    import itertools
    print('Lenghts: ', len(ray_dir[0]), len(ray_pos[0]), len(target_cam_index[0]), len(x_index[0]), len(y_index[0]),
          len(face_index_array[0]), len(x_pixel[0]), len(y_pixel[0]))

    y_or_l_combine = itertools.chain.from_iterable(y_or_l)

    x_or_l_combine = itertools.chain.from_iterable(x_or_l)


    ray_dir_combine = itertools.chain.from_iterable(ray_dir)

    ray_pos_combine = itertools.chain.from_iterable(ray_pos)

    targets_pos = itertools.chain.from_iterable(target_pos)

    targets_cam_index = itertools.chain.from_iterable(target_cam_index)

    face_index_array = itertools.chain.from_iterable(face_index_array)

    x_index = itertools.chain.from_iterable(x_index)

    y_index = itertools.chain.from_iterable(y_index)

    x_pixel = itertools.chain.from_iterable(x_pixel)

    y_pixel = itertools.chain.from_iterable(y_pixel)

    hits = igl.embree.line_mesh_intersection(igl.eigen.MatrixXd(list(ray_pos_combine)),
                                             igl.eigen.MatrixXd(list(ray_dir_combine)), V, F)

    count = 0
    print('Lenghts: ', np.asarray(hits).shape)
    for target, hit, cam_id, x_count, y_count, fac_index, x_im, y_im, x_or, y_or in zip(targets_pos, np.asarray(hits),
                                                                            targets_cam_index, x_index, y_index,
                                                                            face_index_array, x_pixel, y_pixel,
                                                                            x_or_l_combine, y_or_l_combine):

        # print('X and Y: ', x_im, y_im)

        if fac_index == int(hit[0]):
            # if look_up_m[x_im, y_im, 0]!=-1:
            # print('Hit: ', hit)

            # look_up_m[x_count, y_count, 0] = x_im
            # look_up_m[x_count, y_count, 1] = y_im

            look_up_m[x_count, y_count, 0] = x_or
            look_up_m[x_count, y_count, 1] = y_or

            if pano_UV[x_im, y_im, 0] == -1:
                pano_UV[x_im, y_im, 0] = x_count
                pano_UV[x_im, y_im, 1] = y_count

        count += 1

    print('File name: ', filename)
    np.save('../scenes/' + str(scene_id) + '/pano2UV/' + filename, look_up_m)
    prev_idx = None
    for j in range(pano_UV.shape[1]):
        for i in range(pano_UV.shape[0]):
            if int(pano_UV[i, j, 0] != -1 and pano_UV[i, j, 1] != -1):
                # im[i,j] = UV[int(round(m[i,j,0])), int(round(m[i,j,1]))]
                prev_idx = [pano_UV[i, j, 0], pano_UV[i, j, 1]]
                count += 1
                # print('Indexes in UV: ', m[i,j])
            else:
                # Find neightboor i, j

                if j > 0:
                    prev_idx_n = pano_UV[i, j - 1]

                    if prev_idx_n[0] != -1:
                        pano_UV[i, j, 0] = prev_idx_n[0]
                        pano_UV[i, j, 1] = prev_idx_n[1]
                        # im[i, j] = UV[int(round(m[i, j, 0])), int(round(m[i, j, 1]))]

                if j < pano_UV.shape[1] - 1:
                    prev_idx_n = pano_UV[i, j + 1]
                    if prev_idx_n[0] != -1:
                        pano_UV[i, j, 0] = prev_idx_n[0]
                        pano_UV[i, j, 1] = prev_idx_n[1]
                        # im[i, j] = UV[int(round(m[i, j, 0])), int(round(m[i, j, 1]))]

                if i > 0:
                    prev_idx_n = pano_UV[i - 1, j]
                    if prev_idx_n[0] != -1:
                        pano_UV[i, j, 0] = prev_idx_n[0]
                        pano_UV[i, j, 1] = prev_idx_n[1]
                        # im[i, j] = UV[int(round(m[i, j, 0])), int(round(m[i, j, 1]))]

                if i < pano_UV.shape[0] - 1:
                    prev_idx_n = pano_UV[i + 1, j]
                    if prev_idx_n[0] != -1:
                        pano_UV[i, j, 0] = prev_idx_n[0]
                        pano_UV[i, j, 1] = prev_idx_n[1]
                        # im[i, j] = UV[int(round(m[i, j, 0])), int(round(m[i, j, 1]))]

                if prev_idx != None:
                    pano_UV[i, j, 0] = prev_idx[0]
                    pano_UV[i, j, 1] = prev_idx[1]
                    # im[i, j] = UV[int(round(m[i, j, 0])), int(round(m[i, j, 1]))]

    np.save('../scenes/' + str(scene_id) + '/UV2pano/' + filename, pano_UV)

    return look_up_m


def apply_look_up(filepath, pano, out_path):
    m = np.load(filepath)

    im = np.zeros((m.shape[0], m.shape[1], 3))

    count = 0

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            # print(i, j)

            if m[i, j, 0] != -1:  # and m[i,j,1]!=-1):
                print(m[i, j], pano[int(round(m[i, j, 1])), int(round(m[i, j, 0]))])
                im[i, j] = pano[int(round(m[i, j, 1])), int(round(m[i, j, 0]))]
                count += 1
            else:

                if i + 1 < 2048:
                    if m[i + 1, j, 0] != -1:
                        im[i, j] = pano[int(round(m[i + 1, j, 1])), int(round(m[i + 1, j, 0]))]
                        # print('Pano color: ', pano[int(round(m[i + 1, j, 1])), int(round(m[i + 1, j, 0]))], m[i + 1, j, 1], m[i + 1, j, 0])
                elif i - 1 > 0:
                    if m[i - 1, j, 0] != -1:
                        im[i, j] = pano[int(round(m[i - 1, j, 1])), int(round(m[i - 1, j, 0]))]
                elif j + 1 < 2048:
                    if m[i, j + 1, 0] != -1:
                        im[i, j] = pano[int(round(m[i, j + 1, 1])), int(round(m[i, j + 1, 0]))]
                elif i - 1 < 0:
                    if m[i, j - 1, 0] != -1:
                        im[i, j] = pano[int(round(m[i, j - 1, 1])), int(round(m[i, j - 1, 0]))]

    # cv2.imwrite('new_UV_56.png', cv2.flip(cv2.flip(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), 0),1))

    return cv2.flip(cv2.flip(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), 0), 1)

if __name__ == '__main__':

    scene_id = 'google'
    obj_name = 'mesh'

    Vers, Facs, Vuvs, Vns, Fuvs_id, Fns, V, F, TC, FTC = load_mesh()

    for camera_param_file in sorted(os.listdir(path='../scenes/' + str(scene_id) + '/output/RTm/'))[::4]:

        print('Camera file: ', camera_param_file)
        path_to_npy = '../scenes/' + str(scene_id) + '/pano2UV/' + camera_param_file.split('.')[0] + '.npy'
        print('Path: ', path_to_npy)
        if os.path.exists(path_to_npy):
            print('Camera id exist: ', camera_param_file)
            # continue

        idx = int(camera_param_file.split('_')[0])
        img_path = str(idx).zfill(5) + '.png'

        camera_RT_matrices, camera_locations = load_camera_info_np("../scenes/" + str(scene_id) + "/output/RTm/",
                                                                   camera_param_file)

        # Load panoramas
        # R_source, G_source, B_source, A_source = load_panoramas('../Scenes/scene4_clean/output/')
        image = load_panoramas('../scenes/' + str(scene_id) + '/output/', img_path)
        print('Image shape: ', image.shape)

        # UV = cv2.imread('../Scenes/scene4_clean/UVs/reprojection_dense.png')
        # UV = cv2.rotate(UV, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 0)
        # cv2.imshow('test', image)
        # Load mesh

        # cv2.waitKey(0)
        m = look_up_matrix(camera_locations, V, F, Vers, Facs, Vuvs, Fuvs_id, camera_RT_matrices[0],
                           camera_param_file.split('.')[0])
        # m = np.load('look_up_matrix.npy')
        # break
        # pano = apply_look_up('../scenes/'+str(scene_id)+'/pano2UV/'+camera_param_file.split('.')[0]+'.npy', image, '../scenes/'+str(scene_id)+'/'+img_path).astype(np.uint8)
        #
        # cv2.imwrite('test_pano.png', pano)
    # print('Image shape: ', image.shape)
    # # # pSAZZpe) 

    # # cv2.imwrite('result.png', cv2.hconcat([image, pano]))
    # cv2.imshow('Compare', cv2.hconcat([image, pano]))
    # cv2.waitKey(0)
