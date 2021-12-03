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

parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=str,
                    default="",
                    help="Location of obj filee")
parser.add_argument('--start', type=int, default=0, help="UV mapping")
parser.add_argument('--offset', type=int, default=200, help="UV mapping")
parser.add_argument('--visual', type=bool, default=False, help="Display map")
parser.add_argument('--fill_mesh', type=bool, default=True)
parser.add_argument('--create_dict', dest='create_dict', action='store_true')
parser.add_argument('--panoram_size', type=tuple, default=(512, 256))
parser.add_argument('--UV_size', type=tuple, default=(2048, 2048))
parser.add_argument('--scene_number', type=int, default=4)
parser.add_argument('--distance_threshold', type=int, default=20)
parser.add_argument('--processes_number', type=int, default=10)
parser.add_argument('--UV_output_path', type=str, default=10)
parser.add_argument('--step', type=int, default=4)
parser.add_argument('--no_dis_filter', type=bool, default=False)
parser.add_argument('--mesh_name', type=str, default='mesh')
# parser.add_argument('--create_3D_mapping', type=bool, default=False)
parser.add_argument('--create_3D_mapping', dest='create_3D_mapping', action='store_true')
parser.add_argument('--feature', dest='feature', action='store_true')
pars = parser.parse_args()


def load_camera_info_np(directory, start=pars.start, offset=pars.offset):
    camera_RT_matrices = []
    camera_locations = []
    count = 0
    for filename in sorted(os.listdir(directory)):
        if count < start + offset and count >= start and (count % pars.step == 0):
            RTmatrix = np.loadtxt(os.path.join(directory, filename))
            print('RTm: ', filename)
            # Camera rotation and translation matrix
            camera_RT_matrices.append(np.loadtxt(os.path.join(directory, filename)))
            # World coordin<3 location
            camera_locations.append(np.dot(-1 * RTmatrix[:, :-1].T, RTmatrix[:, -1]))
        count += 1
    print('Camera location list: ', len(camera_locations))
    return camera_RT_matrices, camera_locations


def load_panoramas(directory, start=pars.start, offset=pars.offset):
    im_x, im_y = pars.panoram_size

    A_source = igl.eigen.MatrixXuc(im_x, im_y)
    R_source = igl.eigen.MatrixXuc(im_x, im_y)
    G_source = igl.eigen.MatrixXuc(im_x, im_y)
    B_source = igl.eigen.MatrixXuc(im_x, im_y)

    images_R_source = []
    images_G_source = []
    images_B_source = []
    images_A_source = []

    count = 0
    print('Image directory: ', directory)
    for filename in sorted(os.listdir(directory)):
        if filename != 'RTm' and filename != 'white.png':
            count = int(filename.split('.')[0])
            if count < start + offset and count >= start and (count % pars.step == 0):
                print('Panorama names: ', filename)
                igl.png.readPNG(directory + filename, R_source, G_source, B_source, A_source)
                images_R_source.append(np.copy(np.array(R_source)))
                images_G_source.append(np.copy(np.array(G_source)))
                images_B_source.append(np.copy(np.array(B_source)))
                images_A_source.append(np.copy(np.array(A_source)))
        # count+=1

    return images_R_source, images_G_source, images_B_source, images_A_source

def viewer_axis(origin, x, y, z):
    # Create viewer axis
    # Corners of the bounding box
    V_box = igl.eigen.MatrixXd(
        [
            origin,
            x,
            y,
            z,
        ]
    )

    E_box = igl.eigen.MatrixXd(
        [
            [0, 1],
            [0, 2],
            [0, 3],
        ]
    ).castint()

    return V_box, E_box


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# Return only the closest 2
def distance_filtering(camera_positions, centroid, vp1_, vp2_, vp3_, threshold, face_normal):
    min_ = 100000
    max_ = -1
    for index, pos in enumerate(camera_positions):

        # print('Point in distance comparisson: ', centroid, [pos[0], -pos[2], pos[1]], distance(np.array([pos[0], -pos[2], pos[1]]), np.array(centroid)))
        if distance(np.array([pos[0], pos[2], -pos[1]]), np.array(centroid)) < threshold:
            # if index not in camera_indexes:
            if index < min_:
                min_ = index
            if index > max_:
                max_ = index
                # camera_indexes.append(index)

        elif distance(np.array([pos[0], pos[2], -pos[1]]), np.array(vp1_)) < threshold:
            # if index not in camera_indexes:
            if index < min_:
                min_ = index
            if index > max_:
                max_ = index
                # camera_indexes.append(index)

        elif distance(np.array([pos[0], pos[2], -pos[1]]), np.array(vp2_)) < threshold:
            # if index not in camera_indexes:
            if index < min_:
                min_ = index
            if index > max_:
                max_ = index
                # camera_indexes.append(index)

        elif distance(np.array([pos[0], pos[2], -pos[1]]), np.array(vp3_)) < threshold:
            # if index not in camera_indexes:
            if index < min_:
                min_ = index
            if index > max_:
                max_ = index
                # camera_indexes.append(index)

    li = list(range((max_ - min_) + 1))
    camera_indexes = []
    for l in li:
        # print('l_min_: ', l, min_, l+min_)
        camera_indexes.append(l + min_)

    return camera_indexes


def visibility_filtering_UV(camera_locations, V, F, Vers, Facs, distance_threshold, Vuvs, Fuvs_id, R_np):

    print('In visibility filtering...')
    camera_positions = []
    ray_pos = []
    ray_dir = []
    cam_ids = []
    target_pos = []
    target_cam_index = []
    y_index = []
    x_index = []
    face_index_array = []

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

    # Change to per pixel ra
    face_index = 0
    for fac, fac_uv in zip(Facs, Fuvs_id):
        vp1 = Vers[fac[0]]
        vp2 = Vers[fac[1]]
        vp3 = Vers[fac[2]]
        # 		centroid = np.array([(vp1[0]+vp2[0]+vp3[0])/3, (vp1[1]+vp2[1]+vp3[1])/3, (vp1[2]+vp2[2]+vp3[2])/3])

        v1_uvs = Vuvs[int(fac_uv[0])] * R_np.shape
        v2_uvs = Vuvs[int(fac_uv[1])] * R_np.shape
        v3_uvs = Vuvs[int(fac_uv[2])] * R_np.shape

        x_uvs = np.array([v1_uvs[0], v2_uvs[0], v3_uvs[0]])
        y_uvs = np.array([v1_uvs[1], v2_uvs[1], v3_uvs[1]])

        min_index_x = np.argmin(x_uvs)
        max_index_x = np.argmax(x_uvs)
        min_index_y = np.argmin(y_uvs)
        max_index_y = np.argmax(y_uvs)

        x_uvs[min_index_x] -= 1
        x_uvs[max_index_x] += 1
        y_uvs[min_index_y] -= 1
        y_uvs[max_index_y] += 1

        min_x = max(0, int(round(x_uvs[min_index_x])))
        max_x = min(2048, int(round(x_uvs[max_index_x])))

        min_y = max(0, int(round(y_uvs[min_index_y])))
        max_y = min(2048, int(round(y_uvs[max_index_y])))

        # counter_xy = 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                v, w, u = barycentric_coords([x + 0.5, y + 0.5], v1_uvs, v2_uvs, v3_uvs)

                if v >= 0 and w >= 0 and u >= 0:
                    point_3d_coord = compute_3d_coords(vp1, vp2, vp3, w, v, u)
                    for pos, direction, cam_id in zip(camera_positions,
                                                      (np.array(point_3d_coord[:3]) - camera_positions), cam_ids):
                        if distance(camera_positions[cam_id], np.array(point_3d_coord[:3])) <= distance_threshold:
                            # print(pos, direction, cam_id)
                            ray_pos[cam_id].append(pos)
                            ray_dir[cam_id].append(direction)
                            target_pos[cam_id].append(fac)
                            target_cam_index[cam_id].append(cam_id)
                            y_index[cam_id].append(y)
                            x_index[cam_id].append(x)
                            face_index_array[cam_id].append(face_index)
        face_index += 1


    visibility_matrix = np.zeros((len(camera_locations), 2048, 2048), dtype=bool)

    import itertools

    ray_dir_combine = itertools.chain.from_iterable(ray_dir)

    ray_pos_combine = itertools.chain.from_iterable(ray_pos)

    targets_pos = itertools.chain.from_iterable(target_pos)

    targets_cam_index = itertools.chain.from_iterable(target_cam_index)

    face_index_array = itertools.chain.from_iterable(face_index_array)

    x_index = itertools.chain.from_iterable(x_index)

    y_index = itertools.chain.from_iterable(y_index)

    hits = igl.embree.line_mesh_intersection(igl.eigen.MatrixXd(list(ray_pos_combine)),
                                             igl.eigen.MatrixXd(list(ray_dir_combine)), V, F)

    for target, hit, cam_id, x_count, y_count, fac_index in zip(targets_pos, np.asarray(hits), targets_cam_index,
                                                                x_index, y_index, face_index_array):

        if fac_index == int(hit[0]):
            visibility_matrix[cam_id, x_count, y_count] = 1

    np.save(os.path.join(pars.scene_path, "visibility"), visibility_matrix)
    return visibility_matrix


def cast_color_uv(x, y, R_target, G_target, B_target, A_target, c):
    R_target[x, y] = int(c[0])
    G_target[x, y] = int(c[1])
    B_target[x, y] = int(c[2])
    A_target[x, y] = 255


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


def camera_axis(camera_RT):
    # From [X,Y,Z] to [X,-Z,Y]
    origin = l2b([0, 0, 0])
    x_axis = l2b([1, 0, 0])
    y_axis = l2b([0, 1, 0])
    z_axis = l2b([0, 0, 1])

    world_translate = np.dot(-1 * camera_RT[:, :-1].T, camera_RT[:, -1])
    world_translate = np.expand_dims(world_translate, axis=1)

    # Rotation and translation from camera->world coordinates
    origin_camera = np.dot(-1 * camera_RT[:, :-1].T, origin.T) + world_translate
    x_axis_camera = np.dot(-1 * camera_RT[:, :-1].T, x_axis.T) + world_translate
    y_axis_camera = np.dot(-1 * camera_RT[:, :-1].T, y_axis.T) + world_translate
    z_axis_camera = np.dot(-1 * camera_RT[:, :-1].T, z_axis.T) + world_translate

    # From [X,-Z,Y] to [X,Y,Z]
    origin = b2l(origin_camera[:3])
    x_axis = b2l(x_axis_camera[:3])
    y_axis = b2l(y_axis_camera[:3])
    z_axis = b2l(z_axis_camera[:3])

    return origin, x_axis, y_axis, z_axis


def billinear_inter(R_source, G_source, B_source, x, y):
    x_1 = int(np.floor(x))
    x_2 = int(np.ceil(x))

    y_1 = int(np.floor(y))
    y_2 = int(np.ceil(y))

    x_inter = np.array([[x_2 - x, x - x_1]])
    y_inter = np.array([[y_2 - y], [y - y_1]])

    R_matrrix = [[R_source[x_1, y_1], R_source[x_1, y_2]], [R_source[x_2, y_1], R_source[x_2, y_2]]]
    G_matrrix = [[G_source[x_1, y_1], G_source[x_1, y_2]], [G_source[x_2, y_1], G_source[x_2, y_2]]]
    B_matrrix = [[B_source[x_1, y_1], B_source[x_1, y_2]], [B_source[x_2, y_1], B_source[x_2, y_2]]]

    R = np.dot(np.dot(x_inter, R_matrrix), y_inter)
    G = np.dot(np.dot(x_inter, G_matrrix), y_inter)
    B = np.dot(np.dot(x_inter, B_matrrix), y_inter)
    return np.array([R[0], G[0], B[0]])


def length_line(p1, p2):
    return math.sqrt(((p1 ** 2) + (p2 ** 2)))


def magnitude(point_3d):
    return math.sqrt((point_3d[0] ** 2) + (point_3d[1] ** 2) + (point_3d[2] ** 2))


def compute_3d_coords(vp1, vp2, vp3, w, v, u):
    pixel_3d = [vp1[0] * u + vp2[0] * v + vp3[0] * w, vp1[1] * u + vp2[1] * v + vp3[1] * w,
                vp1[2] * u + vp2[2] * v + vp3[2] * w, 1]

    return pixel_3d


def Equirectangular(R_source, G_source, B_source, A_source, point_3d_coord, x_scale, y_scale):
    x_or = ((math.atan2(point_3d_coord[0], - point_3d_coord[2]) + math.pi) / (2 * math.pi))
    x = x_or * (x_scale - 1)
    y_or = ((math.atan2(point_3d_coord[1], length_line(point_3d_coord[2], point_3d_coord[0])) + (
                math.pi / 2)) / math.pi)
    y = y_or * (y_scale - 1)

    return billinear_inter(R_source, G_source, B_source, x, y), x, y, x_or, y_or


def l2b(point_3d):
    return np.array([[point_3d[0], -point_3d[2], point_3d[1], 1]])


def l2b_list(point_3d):
    return [point_3d[0], -point_3d[2], point_3d[1]]


def b2l(point_3d):
    return np.array([point_3d[0], point_3d[2], -point_3d[1]])


def transform_RT_blender(camera_matrix, point_3d):
    # Transform to blender coords
    minus_Z = l2b([point_3d[0], point_3d[1], point_3d[2]])
    # Tranform 3d point to camera coordinats
    minus_Z_camera_coords = np.dot(camera_matrix, minus_Z.T)

    return minus_Z_camera_coords


def average_c(color_list):
    return np.mean(color_list, axis=0)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum()


def closer_distance_c(color_list, dist_list):
    # print("dist list: ", dist_list)
    return color_list[np.argmax(dist_list), :]


def distance_c(color_list, dist_list):
    # Probabilistic distribution
    weights = softmax(dist_list)
    color_list = (color_list.T * weights).T

    return np.sum(color_list, axis=0)


def angle_(v1, v2):
    # print(v1,[v1[0],v1[2]])
    if length_([v1[0], v1[2]]) == 0:
        return 0
    try:
        # print('In angle ', v1, v1, length_([v1[0], v1[2]]))
        return math.acos(np.dot([v1[0], v1[2]], [v2[0], v2[2]]) / (length_([v1[0], v1[2]]) * length_([v2[0], v2[2]])))
    except RuntimeWarning:
        print('v1: ', v1)
        print('v2: ', v2)
        return 0


def length_(v):
    return math.sqrt(np.dot(v, v))


x_dim = pars.UV_size[0]
y_dim = pars.UV_size[1]

# Ceate Shared Arrays
shared_array_base_R = multiprocessing.Array(ctypes.c_int, x_dim * y_dim)
R_array = np.ctypeslib.as_array(shared_array_base_R.get_obj())
R_array = R_array.reshape(x_dim, y_dim)
R_array[:, :] = 211

shared_array_base_G = multiprocessing.Array(ctypes.c_int, x_dim * y_dim)
G_array = np.ctypeslib.as_array(shared_array_base_G.get_obj())
G_array = G_array.reshape(x_dim, y_dim)
G_array[:, :] = 211

shared_array_base_B = multiprocessing.Array(ctypes.c_int, x_dim * y_dim)
B_array = np.ctypeslib.as_array(shared_array_base_B.get_obj())
B_array = B_array.reshape(x_dim, y_dim)
B_array[:, :] = 211

shared_array_base_A = multiprocessing.Array(ctypes.c_int, x_dim * y_dim)
A_array = np.ctypeslib.as_array(shared_array_base_A.get_obj())
A_array = A_array.reshape(x_dim, y_dim)
A_array[:, :] = 255

manager = multiprocessing.Manager()
UV2panos_dict = manager.dict()


def texture_func(visibility_matrix, camera_locations, threshold, Facs, Fuvs_id, face_normals, Vers, Vuvs, R_np,
                 start_index, end_index, R_source, G_source, B_source, A_source, camera_RT_matrices, f, f2,
                 def_param=(R_array, G_array, B_array, A_array, UV2panos_dict)):
    uv_offset = 0
    im_x, im_y = pars.panoram_size

    count = start_index
    path_vector = np.array([camera_locations[-1][0], -camera_locations[-1][1]]) - np.array(
        [camera_locations[0][0], -camera_locations[0][1]])
    length = np.linalg.norm(path_vector)
    path_unit_vector = path_vector / length


    for fac, fac_uv, face_n in tqdm(
            zip(Facs[start_index:end_index], Fuvs_id[start_index:end_index], face_normals[start_index:end_index]),
            total=Facs.shape[0]):

        vp1 = Vers[fac[0]]
        vp2 = Vers[fac[1]]
        vp3 = Vers[fac[2]]

        centroid = [(vp1[0] + vp2[0] + vp3[0]) / 3, (vp1[1] + vp2[1] + vp3[1]) / 3, (vp1[2] + vp2[2] + vp3[2]) / 3]

        active_cameras = distance_filtering(camera_locations, centroid, [vp1[0], vp1[1], vp1[2]],
                                            [vp2[0], vp2[1], vp2[2]], [vp3[0], vp3[1], vp3[2]], threshold, face_n)

        # Scale coordinates uv coordinates
        v1_uvs = Vuvs[int(fac_uv[0])] * R_np.shape
        v2_uvs = Vuvs[int(fac_uv[1])] * R_np.shape
        v3_uvs = Vuvs[int(fac_uv[2])] * R_np.shape

        x_uvs = np.array([v1_uvs[0], v2_uvs[0], v3_uvs[0]])
        y_uvs = np.array([v1_uvs[1], v2_uvs[1], v3_uvs[1]])

        min_index_x = np.argmin(x_uvs)
        max_index_x = np.argmax(x_uvs)
        min_index_y = np.argmin(y_uvs)
        max_index_y = np.argmax(y_uvs)

        x_uvs[min_index_x] -= 1
        x_uvs[max_index_x] += 1
        y_uvs[min_index_y] -= 1
        y_uvs[max_index_y] += 1

        min_x = max(0, int(round(x_uvs[min_index_x])))
        max_x = min(R_np.shape[0], int(round(x_uvs[max_index_x])))
        min_y = max(0, int(round(y_uvs[min_index_y])))
        max_y = min(R_np.shape[0], int(round(y_uvs[max_index_y])))


        for x in range(min_x, max_x):
            for y in range(min_y, max_y):

                v, w, u = barycentric_coords([x + 0.5, y + 0.5], v1_uvs, v2_uvs, v3_uvs)

                string_list = []
                if v + uv_offset >= 0 and w + uv_offset >= 0 and u + uv_offset >= 0:

                    color_list = []
                    distance_list = []
                    angle_list = []

                    # Compute 3D coordinates
                    point_3d_coord = compute_3d_coords(vp1, vp2, vp3, w, v, u)

                    # Get colors from active cameras
                    camera_list = []
                    if len(active_cameras) > 0:
                        f2.write('\nrow:%d,%d,%f,%f,%f' % (x, y, point_3d_coord[0], point_3d_coord[1], point_3d_coord[2]))
                    for camera_id in active_cameras:


                        # if visibility_matrix[camera_id, count] or pars.no_dis_filter:
                        if visibility_matrix[camera_id, x, y] or pars.no_dis_filter:

                            RT_matrix = camera_RT_matrices[camera_id]

                            # Transform point to camera coordinates.
                            cam_point = camera_locations[camera_id]

                            cam_point = np.array([cam_point[0], cam_point[2], -cam_point[1]])

                            dist = distance(cam_point, np.array(point_3d_coord))

                            point_3d_coord_changed = l2b_list(point_3d_coord)
                            cam_point_changed = np.array(l2b_list(cam_point))

                            x_dist = np.sqrt((point_3d_coord_changed[1] - cam_point_changed[1]) ** 2 + (
                                        cam_point_changed[0] - point_3d_coord_changed[0]) ** 2 + (
                                                         cam_point_changed[2] - point_3d_coord_changed[2]) ** 2)

                            v = [cam_point_changed[0] - point_3d_coord_changed[0],
                                 cam_point_changed[1] - point_3d_coord_changed[1]]

                            # print(v)
                            cam_dir_ray = point_3d_coord[:3] - cam_point
                            angle_cam_face = math.degrees(angle_(face_n, -cam_dir_ray))

                            point_3d_cam_coord = transform_RT_blender(RT_matrix, point_3d_coord)

                            color_, x_im_coord, y_im_coord, x_origin, y_origin = Equirectangular(R_source[camera_id],
                                                                                                 G_source[camera_id],
                                                                                                 B_source[camera_id],
                                                                                                 A_source[camera_id],
                                                                                                 point_3d_cam_coord,
                                                                                                 im_x, im_y)

                            string_list.append("\nrow:%d,%.2f,%.2f,%.4f,%.4f,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f" % (
                            camera_id, dist, angle_cam_face, x_origin, y_origin,
                            x_im_coord, y_im_coord, x, y, v[0], v[1], x_dist, cam_point_changed[0],
                            cam_point_changed[1], cam_point_changed[2]))

                            color_list.append(color_)

                            distance_list.append(-x_dist)

                            angle_list.append(-angle_cam_face)

                            camera_list.append(camera_id)

                        # print('In Color List', color_list)
                    if len(color_list):
                        color_array = np.squeeze(np.array(color_list), axis=2)

                        # Blending Approach
                        color = distance_c(color_array, distance_list)

                        # index_argmax = np.argmax(distance_list)
                        index = np.argsort(distance_list)


                        # Compute projection
                        c_p_vector = string_list[index[-1]].split(':')[1].split(',')
                        c_p_vector = np.array([float(c_p_vector[-6]), -float(c_p_vector[-5])])
                        c_p_vector_hat = c_p_vector / np.linalg.norm(c_p_vector)

                        cos_theta = np.dot(c_p_vector_hat, path_unit_vector)

                        projection_magnitude = np.linalg.norm(c_p_vector) * cos_theta

                        f.write(string_list[index[-1]] + ' ,' + str(projection_magnitude))  # +str(percentage_1))

                        string_list = []
                        # Cast color in UV map
                        cast_color_uv(x, y, R_array, G_array, B_array, A_array, color)

        count += 1


def angle_c(color_list, face_normal, cam_directions):
    angles = []
    for cam_dir in cam_directions:
        angles.append(-angle_(face_normal, -cam_dir))

    weights = softmax(angles)

    color_list = (color_list.T * weights).T

    return np.sum(color_list, axis=0)


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
    print('Mesh name: ', pars.mesh_name)
    igl.readOBJ(pars.scene_path + '/' + pars.mesh_name + '.obj', V, TC, CN, F, FTC, FN)

    # Store information as numpy array
    Vers = np.asarray(V)
    Facs = np.asarray(F)
    Vuvs = np.asarray(TC)
    Vns = np.asarray(CN)
    Fuvs_id = np.asarray(FTC)
    Fns = np.asarray(FN)

    # Compute NOrmals
    igl.per_face_normals(V, F, PFN)
    face_normals = np.asarray(PFN)

    return Vers, Facs, Vuvs, Vns, Fuvs_id, Fns, face_normals, V, F, TC, FTC


def main(Vers, Facs, Vuvs, Vns, Fuvs_id, Fns, face_normals):
    V_box = []

    camera_RT_matrices, camera_locations = load_camera_info_np(pars.scene_path + "/images/RTm")

    # Load panoramas
    R_source, G_source, B_source, A_source = load_panoramas(pars.scene_path + '/images/')

    # Target output
    R_target = igl.eigen.MatrixXuc(x_dim, y_dim) * 0
    G_target = igl.eigen.MatrixXuc(x_dim, y_dim) * 0
    B_target = igl.eigen.MatrixXuc(x_dim, y_dim) * 0
    A_target = igl.eigen.MatrixXuc(x_dim, y_dim) * 0
    # A_target2 = igl.eigen.MatrixXuc(x_dim, y_dim)*0

    R_np = np.asarray(R_target)

    # camera = camera_locations[0]

    if (os.path.exists(os.path.join(pars.scene_path, "visibility.npy"))):
        visibility_matrix = np.load(os.path.join(pars.scene_path, "visibility.npy"))
    else:
        visibility_matrix = visibility_filtering_UV(camera_locations, V, F, Vers, Facs, pars.distance_threshold,
                                                 Vuvs, Fuvs_id, R_np)


    print('Visibility matrix is ready..')

    processes = []
    number_of_faces_per_process = int(Facs.shape[0] / pars.processes_number)

    if os.path.exists("points.txt"):
        os.remove("points.txt")

    f = open("points.txt", "a")

    f2 = open("points23D.txt", "a")


    # start = time.time()
    UV_to_3D = np.zeros((2048, 2048, 3)) - 1

    print('Before texturing loop')
    for p_id in range(pars.processes_number):
        p = multiprocessing.Process(target=texture_func,
                                    args=[visibility_matrix, camera_locations, pars.distance_threshold,
                                          Facs, Fuvs_id, face_normals, Vers, Vuvs, R_np,
                                          number_of_faces_per_process * p_id, number_of_faces_per_process * (p_id + 1),
                                          R_source, G_source, B_source, A_source, camera_RT_matrices, f, f2])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    f.close()
    f2.close()

    np.save('1', UV_to_3D)

    uv_img = np.concatenate((np.expand_dims(B_array, axis=2), np.expand_dims(G_array, axis=2),
                             np.expand_dims(R_array, axis=2), np.expand_dims(A_array, axis=2)), axis=2)

    uv_path = pars.scene_path + '/UVs/reprojection.png'
    cv2.imwrite(uv_path, cv2.flip(cv2.flip(cv2.rotate(uv_img, cv2.ROTATE_90_CLOCKWISE), 1), 0))

    if pars.visual == True:
        for i in range(R_array.shape[0]):
            for j in range(R_array.shape[1]):
                R_target[i, j] = R_array[i, j]
                G_target[i, j] = G_array[i, j]
                B_target[i, j] = B_array[i, j]
                A_target[i, j] = A_array[i, j]

    igl.png.writePNG(R_target, G_target, B_target, A_target, pars.scene_path + '/UVs/orig_UV.png')

    for index, camera in enumerate(camera_locations):
        V_box.append([camera[0], camera[2], -camera[1]])

    return R_target, G_target, B_target, V_box, uv_path


def visualize_textured_mesh(R_target, G_target, B_target, V, F, TC, FTC):
    v_color = igl.eigen.MatrixXd([[1, 1, 1]])
    # Viewer
    viewer = igl.glfw.Viewer()

    # Set mesh
    viewer.data(0).set_mesh(V, F)

    # Set uv map
    viewer.data(0).set_texture(R_target, G_target, B_target)

    # Set uv vertices coordinates
    viewer.data(0).set_uv(TC, FTC)

    # Set vertices color to 1
    viewer.data(0).set_colors(v_color)

    viewer.data(0).show_texture = True
    viewer.launch()


if __name__ == '__main__':
    #
    Vers, Facs, Vuvs, Vns, Fuvs_id, Fns, face_normals, V, F, TC, FTC = load_mesh()

    R_target, G_target, B_target, V_box, uv_path = main(Vers, Facs, Vuvs, Vns, Fuvs_id, Fns, face_normals)

    print('Create Dictionaries: ', pars.create_dict)
    if pars.create_dict:
        print('Creating dictionary..')
        dict_.dictionary('points.txt', pars.scene_path)

    # Create UV_to_3D_mapping
    print('Create 3D mapping: ', pars.create_3D_mapping)
    if pars.create_3D_mapping:
        UV_to_3D = np.zeros((2048, 2048, 3)) - 1

        f = open('points23D.txt', "r")
        for idx, line in enumerate(tqdm(f)):
            if idx != 0:

                line = line.split(':')[1].split(',')
                UV_to_3D[int(line[0]), int(line[1]), :] = np.array([float(line[2]), float(line[3]), float(line[4])])

        np.save(pars.scene_path+pars.mesh_name, UV_to_3D)
