import os
import cv2
import json
from tqdm import tqdm



def dictionary(source_file, out_dir):

	f=open(source_file, "r")

	UV2panos_dict = {}

	frames_dictionary = {}

	for line in tqdm(f):

		if line[:3]=="row":
			# print("line : ", count, line.strip().split(":")[1].split(','))

			entries = line.strip().split(":")[1].split(',')
			# print(entries)
								
			frame_id = int(entries[0])

			distance = float(entries[11])
			# print('Distances: ', entries[1], entries[11])
			angle = float(entries[2])

			x_p = entries[5]
			y_p = entries[6]

			x_p_or = entries[3]
			y_p_or = entries[4]
			
			x_UV = int(entries[7])
			y_UV = int(entries[8])

			camera_dist = entries[9]

			path_dist = entries[-1]

			# print("entrie: ", entries)#, entries[-1][:5])

			if str(frame_id) not in frames_dictionary.keys():
				frames_dictionary[str(frame_id)] = {}
				dict_ = frames_dictionary[str(frame_id)]
			else:
				dict_ = frames_dictionary[str(frame_id)]

			if str(x_p)+","+str(y_p) not in dict_.keys(): 
				dict_[str(x_p)+","+str(y_p)] = []
				dict_[str(x_p)+","+str(y_p)].append([x_UV, y_UV, x_p_or, y_p_or, distance, float(entries[-1][:4]), float(path_dist)])
			else:
				dict_[str(x_p)+","+str(y_p)].append([x_UV, y_UV, x_p_or, y_p_or, distance, float(entries[-1][:4]), float(path_dist)])

	# print(frames_dictionary['5'])
	for key in frames_dictionary:
		with open(out_dir+'/pano2UV_proj/'+key+'_dict.json', 'w') as f:
			json.dump(frames_dictionary[key], f)
	#
	#
	# with open(out_dir+'/frames_dict.json', 'w') as f:
		# json.dump(frames_dictionary, f)
