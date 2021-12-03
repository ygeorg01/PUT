import cv2
import numpy as np

UV = cv2.imread('../../Scenes/1/UVs/reprojection.png')

UV_dense = UV.copy()
print('UV shape: ', UV.shape)
for i in range(UV.shape[0]):
	for j in range(UV.shape[1]):
		# break
		# print(UV[i,j])
		if (UV[i,j] == [211, 211, 211]).all():
			colors = []
			for offset_i in [i-1, i, i+1]:
				for offset_j in [j-1, j, j+1]:
					if not(offset_j>=2048 or offset_j<0 or offset_i>=2048 or offset_i<0):
						if (UV[offset_i, offset_j] != [211, 211, 211]).any():
							colors.append(UV[offset_i, offset_j])

			if len(colors)!=0:
				# print('Colors: ', np.array(colors), np.mean(np.array(colors), axis=0), UV[i,j])
				UV_dense[i,j] = np.mean(np.array(colors))


cv2.imwrite('../../Scenes/1/UVs/reprojection_dense.png', UV_dense)