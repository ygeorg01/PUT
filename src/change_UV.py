import bpy
import sys
# import numpy as np
# import cv2
from os.path import isfile
 
imgPath = sys.argv[-1]

# /home/visual-computing-1/Desktop/projects/texturing/code/utils/
scene = bpy.data.scenes["Scene"]

print("Image path: ", imgPath)
img2 = bpy.data.images.load(imgPath)

m = bpy.data.materials['Material']

# scene.camera.location.x = c[3]
# scene.camera.location.y = c[4]
# scene.camera.location.z = c[5]

# img2 = bpy.data.images.load("../code/results/10_UVs/UV_map_"+str(count)+".png")
for k,v in m.node_tree.nodes.items():
    print('In change UV: ', k)
    if k == 'Image Texture':
        print("Changing texture!!")
        v.image=img2