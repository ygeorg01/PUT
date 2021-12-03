import os
from utils.update_UV import update_uv
import argparse
import cv2
import sys
import time
import torch
import json
import numpy as np

# Load Initial UV texture map normal
def load_UV(UV_path):
    return cv2.imread(UV_path, cv2.IMREAD_UNCHANGED)


# Get new consistency image
def create_consistency_and_mask(frame_number, UV_path, blender_file, consistency_out_dir):
    os.system("blender %s -b -P change_UV.py --render-output %s/##### --render-frame %d  -- pathToImage %s" % (
    blender_file, consistency_out_dir, frame_number, UV_path))


# blender consistency_script2.blend -b -P ../code/change_UV.py --render-output ../code/blender_consistency/ --render-frame 14 -- pathToImage /home/visual-computing-1/Desktop/projects/texturing/code/utils/UV_map_13.png
def generator_net(frame_number, dataroot, model, output_dir):
    # Run generator
    testing_command = "python test.py --dataroot  %s%s.png --name %s --model cut_con --batch_size 1 --results_dir %s --no_dropout --num_test 1 --input_nc 3 --dataset_mode single" % (
    dataroot, str(frame_number).zfill(5), model, output_dir)

    # print('Dataroot: ', dataroot)
    # print('testing command: ', testing_command)
    os.system(testing_command)

# Texture
def texture(pars):

    # Set paths
    consistency_path = os.path.join(pars.scene_path, pars.consistency_path)
    gen_output_path = os.path.join(pars.scene_path, pars.output_path)
    dictionary_path = os.path.join(pars.scene_path, pars.dictionary_path)
    UV_folder_path = os.path.join(pars.scene_path, pars.UVs_path)
    blend_file_path = os.path.join(pars.scene_path, pars.blender_file_path)
    render_path = os.path.join(pars.scene_path, pars.render_path)
    unrender_path = os.path.join(pars.scene_path, pars.unrender_path)

    torch.set_grad_enabled(False)

    frame_list = []
    for i in range(pars.start_frame, pars.end_frame, pars.step):
        frame_list.append(i)

    print('Start texturing..')
    start_time = time.time()
    UV_path = UV_folder_path

    UV_colors = {}
    # Create the first image

    for count, frame_number in enumerate(frame_list):
        # break
        start = time.time()

        # Create consistency image
        create_consistency_and_mask(frame_number, UV_path+'/UV_'+str(frame_number).zfill(5)+'.png', blend_file_path, consistency_path)

        # Run Generator
        generator_net(frame_number, consistency_path, pars.model_name, gen_output_path)

        # Update UV map
        print('UV map path: ', UV_path+'/UV_'+str(frame_number).zfill(5)+'.png')
        # print('UV_map: ', UV_map.shape)
        # print('Path to UV update pack: ', gen_output_path+str(frame_number).zfill(5)+".png")
        print('Count: ', count)
        if count == 0 or pars.blend=='no_blend':
            UV_map = cv2.imread(UV_path + '/UV_' + str(frame_number).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)
            update_uv(gen_output_path + str(frame_number).zfill(5) + ".png", frame_number, dictionary_path,
                      UV_folder_path, UV_map, (512, 256), pars.step, UV_colors, True)
        # # else:ls
        # UV_map = cv2.flip(UV_map, 0)
        elif pars.blend=='average':
            UV_map = cv2.imread(UV_path + '/UV_' + str(frame_number).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)[:, :,
                     [2, 1, 0, 3]]
            average_blend(frame_number, gen_output_path, dictionary_path, render_path, unrender_path, UV_map,
                         pars.step, UV_folder_path, pars.scene_path)

        elif pars.blend=='custom':
            print('generated out path: ', gen_output_path)
            UV_map = cv2.imread(UV_path + '/UV_' + str(frame_number).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)[:, :,
                     [2, 1, 0, 3]]
            custom_blend(frame_number, gen_output_path, dictionary_path, render_path, unrender_path, UV_map,
                         pars.step, UV_folder_path, pars.scene_path)

        stop = time.time()
        duration = stop - start
        print(str(duration) + ': For iter :' + str(count))

        torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print('Time: ', elapsed_time / 60)


def custom_blend(frame_id, panorama_path, dictionary_path, render_path, unrender_path, prev_UV_map, step,
                 UV_path, scene_path):

    current_proj = load_dictionary(dictionary_path + '/' + str(int(round((frame_id) / step))) + '_dict.json')

    pano2UV_current = np.load(os.path.join(unrender_path, str(frame_id).zfill(6) + '_RTm.npy'))


    pano2UV_prev = np.load(os.path.join(unrender_path, str(frame_id - step).zfill(6) + '_RTm.npy'))

    prev_pano = cv2.imread(os.path.join(panorama_path, str(frame_id - step).zfill(5) + '.png'))
    current_pano = cv2.imread(os.path.join(panorama_path, str(frame_id).zfill(5) + '.png'))
    # print('Prev pano: ', prev_pano)
    mapping_3D = np.load(os.path.join(scene_path, '005.npy'))

    camera_prev = np.loadtxt(os.path.join(scene_path, 'output', 'RTm', str(frame_id - step).zfill(6) + '_RTm.txt'))
    camera_prev_loc = np.dot(-1 * camera_prev[:, :-1].T, camera_prev[:, -1])
    camera_prev_loc = np.array([camera_prev_loc[0], camera_prev_loc[2], -camera_prev_loc[1]])

    camera_current = np.loadtxt(os.path.join(scene_path, 'output', 'RTm', str(frame_id).zfill(6) + '_RTm.txt'))
    camera_current_loc = np.dot(-1 * camera_current[:, :-1].T, camera_current[:, -1])
    camera_current_loc = np.array([camera_current_loc[0], camera_current_loc[2], -camera_current_loc[1]])

    path_vector = np.array([camera_prev_loc[0], camera_prev_loc[2]]) - np.array(
        [camera_current_loc[0], camera_current_loc[2]])
    length = np.linalg.norm(path_vector)
    cam_dist = np.linalg.norm(camera_current_loc - camera_prev_loc)

    path_unit_vector = path_vector / length

    prev_UV = build_UV_map(cv2.flip(prev_pano, 0), pano2UV_prev, os.path.join(UV_path, str(frame_id - step) + 'build.png'))

    prev_UV = cv2.rotate(prev_UV[:, :, [2, 1, 0, 3]], cv2.ROTATE_90_CLOCKWISE)
    current_UV = build_UV_map(cv2.flip(current_pano, 0), pano2UV_current,
                              os.path.join(UV_path, str(frame_id) + 'build.png'))
    current_UV = cv2.rotate(current_UV[:, :, [2, 1, 0, 3]], cv2.ROTATE_90_CLOCKWISE)

    mapping_2D = []
    paired_points_prev = []
    mapping_2D_seen_prev = []
    weights = []

    mapping_2D_seen_current = []
    for l in current_proj.values():
        for j in l:
            # print('Camera locations: ', camera_prev_loc[[0, 2]], camera_current_loc[[0, 2]])
            c_p_vector = camera_prev_loc[[0, 2]] - mapping_3D[j[0], j[1]][[0, 2]]

            c_p_vector_hat = c_p_vector / np.linalg.norm(c_p_vector)

            cos_theta = np.dot(c_p_vector_hat, path_unit_vector)

            projection_magnitude_1 = np.linalg.norm(c_p_vector) * cos_theta

            c_p_vector = camera_current_loc[[0, 2]] - mapping_3D[j[0], j[1]][[0, 2]]

            c_p_vector_hat = c_p_vector / np.linalg.norm(c_p_vector)

            cos_theta = np.dot(c_p_vector_hat, path_unit_vector)

            projection_magnitude_2 = -np.linalg.norm(c_p_vector) * cos_theta

            if  (projection_magnitude_1+projection_magnitude_2 < cam_dist+0.1) \
                    and prev_UV[j[0], j[1], 3]!=0 and projection_magnitude_1<cam_dist*1.6 \
                        and projection_magnitude_1>(cam_dist/2)-0.01:

                mapping_2D.append([j[0], j[1]])
                dist1 = abs(projection_magnitude_1)-(cam_dist/2)
                dist2 = cam_dist - dist1

                weights_point = np.array([-dist1, -dist2])
                weights_point = np.exp(weights_point - max(weights_point))
                weights_point = weights_point  / weights_point.sum()

            else:
                mapping_2D.append([j[0], j[1]])
                weights_point = np.array([0, 1])

            weights.append([weights_point[0], weights_point[1]])

    mapping_2D = np.array(mapping_2D)

    new_UV = cv2.rotate(prev_UV_map, cv2.ROTATE_90_CLOCKWISE)

    weights = np.array(weights)

    new_UV[mapping_2D[:, 0], mapping_2D[:, 1], :3] = prev_UV[mapping_2D[:, 0], mapping_2D[:, 1], :3] * np.expand_dims(weights[:, 0], axis=1)  \
                                                     + current_UV[mapping_2D[:, 0], mapping_2D[:, 1] , :3] * np.expand_dims(weights[:, 1], axis=1)
    # new_UV[mapping_2D[:, 0], mapping_2D[:, 1], :3] = [255, 0, 0]
    new_UV[mapping_2D[:, 0], mapping_2D[:, 1], 3] = 255

    new_UV = cv2.rotate(new_UV[:, :, [2, 1, 0, 3]], cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(UV_path + "/UV_" + str(frame_id + step).zfill(5) + ".png", new_UV)

def average_blend(frame_id, panorama_path, dictionary_path, render_path, unrender_path, prev_UV_map, step,
                 UV_path, scene_path):

    prev_proj = load_dictionary(dictionary_path + '/' + str(int(round((frame_id) / step)) - 1) + '_dict.json')
    current_proj = load_dictionary(dictionary_path + '/' + str(int(round((frame_id) / step))) + '_dict.json')

    pano2UV_current = np.load(os.path.join(unrender_path, str(frame_id).zfill(6) + '_RTm.npy'))

    pano2UV_prev = np.load(os.path.join(unrender_path, str(frame_id - step).zfill(6) + '_RTm.npy'))

    prev_pano = cv2.imread(os.path.join(panorama_path, str(frame_id - step).zfill(5) + '.png'))
    current_pano = cv2.imread(os.path.join(panorama_path, str(frame_id).zfill(5) + '.png'))

    camera_prev = np.loadtxt(os.path.join(scene_path, 'output', 'RTm', str(frame_id - step).zfill(6) + '_RTm.txt'))
    camera_prev_loc = np.dot(-1 * camera_prev[:, :-1].T, camera_prev[:, -1])
    camera_prev_loc = np.array([camera_prev_loc[0], camera_prev_loc[2], -camera_prev_loc[1]])
    # print('Camera loc: ', camera1_loc)

    camera_current = np.loadtxt(os.path.join(scene_path, 'output', 'RTm', str(frame_id).zfill(6) + '_RTm.txt'))
    camera_current_loc = np.dot(-1 * camera_current[:, :-1].T, camera_current[:, -1])
    camera_current_loc = np.array([camera_current_loc[0], camera_current_loc[2], -camera_current_loc[1]])

    path_vector = np.array([camera_prev_loc[0], camera_prev_loc[2]]) - np.array(
        [camera_current_loc[0], camera_current_loc[2]])
    length = np.linalg.norm(path_vector)
    cam_dist = np.linalg.norm(camera_current_loc - camera_prev_loc)

    # print('Vector length: ', length)
    path_unit_vector = path_vector / length

    prev_UV = build_UV_map(cv2.flip(prev_pano, 0), pano2UV_prev, os.path.join(UV_path, str(frame_id - 4) + 'build.png'))
    prev_UV = cv2.rotate(prev_UV[:, :, [2, 1, 0, 3]], cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('prev'+str(frame_id - 4)+'.png', prev_UV)
    # print('Previous UV: ', prev_UV)
    current_UV = build_UV_map(cv2.flip(current_pano, 0), pano2UV_current,
                              os.path.join(UV_path, str(frame_id) + 'build.png'))
    current_UV = cv2.rotate(current_UV[:, :, [2, 1, 0, 3]], cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('current'+str(frame_id)+'.png', current_UV)
    # print('current UV: ', current_UV)
    mapping_2D = []
    paired_points_prev = []

    for l in prev_proj.values():
        for j in l:

            mapping_2D.append([j[0], j[1]])

    for l in current_proj.values():
        for j in l:

            mapping_2D.append([j[0], j[1]])


    mapping_2D = np.array(mapping_2D)

    new_UV = cv2.rotate(prev_UV_map, cv2.ROTATE_90_CLOCKWISE)

    new_UV[mapping_2D[:, 0], mapping_2D[:, 1], :3] = (prev_UV[mapping_2D[:, 0], mapping_2D[:, 1], :3] + current_UV[mapping_2D[:, 0], mapping_2D[:, 1] , :3])\
                                                     / 2
    for map_idx in mapping_2D:
        if (prev_UV[map_idx[0], map_idx[1], 3] != 0
                or current_UV[map_idx[0], map_idx[1], 3] != 0):
            new_UV[map_idx[0], map_idx[1], 3] = 255
    new_UV = cv2.rotate(new_UV[:, :, [2, 1, 0, 3]], cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(UV_path + "/UV_" + str(frame_id + step).zfill(5) + ".png", new_UV)

def build_UV_map(pano, m, out_path):
    im = np.zeros((m.shape[0], m.shape[1], 4)) + 127.5
    # print('In build UV m: ', m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):

            if m[i, j, 0] != -1:  # and m[i,j,1]!=-1):
                # print('in if: ', m[i,j,0])
                r, g, b, x_1, x_2, y_1, y_2 = billinear_inter(pano, m[i, j, 1], m[i, j, 0])
                # im2[i, j] = pano[int(round(m[i + 1, j, 1])), int(round(m[i + 1, j, 0]))]
                # print('Color: ', [r[0], g[0], b[0]])
                im[i, j, :3] = [r[0], g[0], b[0]]
                im[i, j, 3] = 255
                # print('Final colors: ', r, g, b, int(round(m[i,j,1])), int(round(m[i,j,0])))

    return cv2.flip(cv2.flip(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), 0), 1)


def billinear_inter(panorama, x, y):
    scale_x, scale_y, _ = panorama.shape

    x *= (scale_x - 1)
    y *= (scale_y - 1)

    # print('x, y:  ', x, y)

    x_1 = min([int(np.floor(x + 0.00001)), scale_x - 1])
    x_2 = min([int(np.ceil(x + 0.00001)), scale_x - 1])

    if x_1 == scale_x - 1 and x_2 == scale_x - 1:
        x_2 -= 1

    y_1 = min([int(np.floor(y + 0.00001)), scale_y - 1])
    y_2 = min([int(np.ceil(y + 0.00001)), scale_y - 1])

    if y_1 == scale_y - 1 and y_2 == scale_y - 1:
        y_2 -= 1

    x_inter = np.array([[x_2 - x, x - x_1]])
    y_inter = np.array([[y_2 - y], [y - y_1]])

    R_matrrix = [[panorama[x_1, y_1, 0], panorama[x_1, y_2, 0]], [panorama[x_2, y_1, 0], panorama[x_2, y_2, 0]]]
    G_matrrix = [[panorama[x_1, y_1, 1], panorama[x_1, y_2, 1]], [panorama[x_2, y_1, 1], panorama[x_2, y_2, 1]]]
    B_matrrix = [[panorama[x_1, y_1, 2], panorama[x_1, y_2, 2]], [panorama[x_2, y_1, 2], panorama[x_2, y_2, 2]]]

    R = np.dot(np.dot(x_inter, R_matrrix), y_inter)

    G = np.dot(np.dot(x_inter, G_matrrix), y_inter)
    B = np.dot(np.dot(x_inter, B_matrrix), y_inter)

    return R.round(), G.round(), B.round(), x_1, x_2, y_1, y_2


def load_dictionary(dictionary_path):
    with open(dictionary_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define paths
    parser.add_argument('--UVs_path', type=str, default="UVs", help="Location of obj filee")
    parser.add_argument('--grey_pano_path', type=str, default="images/", help="Gray panorama folder name")
    parser.add_argument('--dictionary_path', type=str, default='pano2UV_proj', help='Dictionary name')
    parser.add_argument('--render_path', type=str, default='UV2pano', help='Dictionary name')
    parser.add_argument('--unrender_path', type=str, default='pano2UV', help='Dictionary name')
    parser.add_argument('--scene_path', type=str, default=None, help='Loacation of Scene folder')
    parser.add_argument('--consistency_path', type=str, default='consistency/', help='Consistency folder name')
    parser.add_argument('--output_path', type=str, default='output/', help='Consistency folder name')
    # parser.add_argument('--sem_output_path', type=str, default = 'sem_output/', help='Consistency folder name')
    parser.add_argument('--blender_file_path', type=str, default='blender_con.blend')
    parser.add_argument('--model_path', type=str, default='checkpoints', help='Model path')
    parser.add_argument('--model_name', type=str, default='consistency', help='model type')
    parser.add_argument('--blend', type=str, default=['no_blend, average, neural, custom'])

    # Texturing properties
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=200)
    parser.add_argument('--random_order', type=bool, default=False)
    parser.add_argument('--step', type=int, default=4, help='Define step size')

    # Blend Net params
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.1)

    pars = parser.parse_args()

    print('pars path: ', pars.scene_path)
    texture(pars)
