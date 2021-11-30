# Projective Urban Texturing

## Setup the experiment
* Clone the repository
* Download libigl and example scence data.
* Unzip inside PUT folder
* Run:
<code> python texture_mesh.py --scene_path ../scenes/005 --model_name consistency --blend custom </code>


## Extract Dictionaries and 2D-3D correspondances.
* Run blender script to extract viewpoints matrices.
* Run smartUV unwrap and then extract the mesh as .obj.
* Run <code> python reprojection --scene_path ../scenes/005/ --mesh_name 005 --create_dict --create_3D_mapping </code>
