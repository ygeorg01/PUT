import os
import cv2
import json
from tqdm import tqdm

f=open("info.txt", "r")

UV2panos_dict = {}

frames_dictionary = {}

count = 0
for line in tqdm(f):

	if line[:3]=="row":
		print("line : ", count, line)

		entries = line.strip().split(":")[1].split(',')
		# print(entries)
							
		frame_id = int(entries[0])

		distance = int(float(entries[1]))

		normal = int(float(entries[2]))

		x_p = int(entries[3])
		y_p = int(entries[4])

		x_UV = int(entries[5])
		y_UV = int(entries[6])

		# print("entrie: ", frame_id, distance, normal, x_p, y_p, x_UV, y_UV)

		if str(frame_id) not in frames_dictionary.keys():
			frames_dictionary[str(frame_id)] = {}
			dict_ = frames_dictionary[str(frame_id)]
		else:
			dict_ = frames_dictionary[str(frame_id)]

		if str(x_p)+","+str(y_p) not in dict_.keys(): 
			dict_[str(x_p)+","+str(y_p)] = []
			dict_[str(x_p)+","+str(y_p)].append([x_UV, y_UV, distance, normal])
		else:
			dict_[str(x_p)+","+str(y_p)].append([x_UV, y_UV, distance, normal])

		count+=1
with open('frames_dict.json', 'w') as f:
	json.dump(frames_dictionary, f)
	