import blenderproc as bproc
import bpy
import argparse
import os
import numpy as np
import random

import h5py
import matplotlib.pyplot as plt

from PIL import Image
import glob
import re

def convert_hdf5_to_png(hdf5_path):
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as file:
        # Assuming there's only one dataset in the file
        print('keys',file.keys())
        dataset = file['colors']

        # Convert dataset to a NumPy array
        data = dataset[:]

        # Plot and save as PNG
        plt.imshow(data, cmap='viridis')  # Adjust colormap as needed
        plt.axis('off')  # Turn off axis
        png_path = os.path.splitext(hdf5_path)[0] + '.png'
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to release memory

        """depth_dataset = file['depth']
        # Convert dataset to a NumPy array
        depth_data = depth_dataset[:]
        # Save as .npy file

        # Normalize depth data to range [0, 1] (if not already)
        depth_data_normalized = (depth_data - np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))

        # Scale to uint8 range [0, 255]
        depth_data_uint8 = (depth_data_normalized * 255).astype(np.uint8)

        # Save as JPEG file using PIL
        jpg_path = os.path.splitext(hdf5_path)[0] + '.jpg'
        Image.fromarray(depth_data_uint8).save(jpg_path)


        npy_path = os.path.splitext(hdf5_path)[0] + '.npy'
        print('depth_data',depth_data.shape)
        np.save(npy_path, depth_data)"""

        depth_dataset = file['distance']
        # Convert dataset to a NumPy array
        depth_data = depth_dataset[:]
        # Save as .npy file

        # Normalize depth data to range [0, 1] (if not already)
        depth_data_normalized = (depth_data - np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))

        # Scale to uint8 range [0, 255]
        depth_data_uint8 = (depth_data_normalized * 255).astype(np.uint8)

        # Save as JPEG file using PIL
        jpg_path = os.path.splitext(hdf5_path)[0] + '.jpg'
        Image.fromarray(depth_data_uint8).save(jpg_path)

        npy_path = os.path.splitext(hdf5_path)[0] + '.npy'
        print('depth_data', depth_data.shape)
        np.save(npy_path, depth_data)

def process_hdf5_files(folder_path):
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.hdf5'):
            hdf5_path = os.path.join(folder_path, filename)
            convert_hdf5_to_png(hdf5_path)


parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('cc_material_path', nargs='?', default="resources/cctextures", help="Path to CCTextures folder, see the /scripts for the download script.")
parser.add_argument("output_dir", nargs='?', default="scenes/", help="Path to where the data should be saved")

args = parser.parse_args()

print('front',args.front)
scene_ui = args.front.split('/')[-1].split('.')[0]
base_path= os.path.join( args.output_dir, scene_ui)
os.makedirs(base_path, exist_ok=True)
output_dir = os.path.join( base_path, 'geom.ply')
print('ouputdir',output_dir)


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else -1

"""# Generate a list of paths for all pose files in the directory and sort them numerically
pose_files = sorted(glob.glob(os.path.join(args.poses_dir, "*.txt")), key=lambda x: extract_number(os.path.basename(x)))

print(pose_files)
"""
if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
for floor in floors:
    # For each material of the object
    for i in range(len(floor.get_materials())):
        # In 95% of all cases
        if np.random.uniform(0, 1) <= 0.95:
            # Replace the material with a random one
            print("Loaded materials:", cc_materials)
            print("Number of loaded materials:", len(cc_materials))
            floor.set_material(i, random.choice(cc_materials))


baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
for obj in baseboards_and_doors:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # Replace the material with a random one
        obj.set_material(i, random.choice(wood_floor_materials))


walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
for wall in walls:
    # For each material of the object
    for i in range(len(wall.get_materials())):
        # In 50% of all cases
        if np.random.uniform(0, 1) <= 0.3:
            # Replace the material with a random one
            wall.set_material(i, random.choice(marble_materials))

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects, amount_of_objects_needed_per_room=1)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

poses = 0
tries = 0

# Prepare to join all objects
bpy.ops.object.select_all(action='DESELECT')
for obj in loaded_objects:
    obj.blender_obj.select_set(True)

# Join all selected objects into one mesh
bpy.context.view_layer.objects.active = loaded_objects[0].blender_obj
bpy.ops.object.join()

# Get the joined object (assuming only one mesh is left selected)
joined_obj = bpy.context.active_object


# Export the joined object as a PLY file
bpy.ops.export_mesh.ply(filepath=output_dir)

"""def check_name(name):
    #for category_name in ["chair", "sofa", "table", "bed", "lamp", "cabinet"]:
    for category_name in ["bed"]:
        if category_name in name.lower():
            return True
    return False

# filter some objects from the loaded objects, which are later used in calculating an interesting score
print([obj.get_name() for obj in loaded_objects])
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]


t= np.array([[1, 0, 0, -6.83],
[0, 1, 0, -5.28],
[0, 0, 1, -0.03],
[0.000000, 0.000000, 0.000000, 1.000000]])

t_p = np.array([[-0.991876, -0.109951, -0.063975, 1.956688],
[0.108240, -0.993683, 0.029644, 2.297590],
[-0.066830, 0.022479, 0.997511, -0.389082],
[0.000000, 0.000000, 0.000000, 1.000000]])

t_s =[[-0.793501, -0.087961, -0.051180, 1.000248],
[0.086592, -0.794946, 0.023715, 1.368025],
[-0.053464, 0.017983, 0.798009, -0.146885],
[0.000000, 0.000000, 0.000000, 1.000000]]





K = bproc.camera.get_intrinsics_as_K_matrix()
print(K)

K[0, 0] /= 2
K[1, 1] /= 2
K[0, 2] = 184.5
K[1, 2] = 184.5
print(K)
bproc.camera.set_intrinsics_from_K_matrix(K, 369, 369)

proximity_checks = {"min": 0.25, "avg": {"min": 0.25, "max": 12.5}, "no_background": False}
for pose in pose_files:    # Sample point inside house
    print(pose)
    cam2world_matrix = np.loadtxt(pose)

    cam2world_matrix[:3, 2] = -cam2world_matrix[:3, 2]
    bproc.camera.add_camera_pose(cam2world_matrix)
    poses += 1
    tries += 1

# Also render normals
bproc.renderer.enable_normals_output()
bproc.renderer.enable_distance_output(False)
#bproc.renderer.enable_depth_output(False)
bproc.renderer.enable_segmentation_output(map_by=["category_id"])

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)

process_hdf5_files(args.output_dir)
"""