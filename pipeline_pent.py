import os
import subprocess
import sys
import shutil
import argparse


def delete_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        print(f'Directory {directory_path} and all its contents have been deleted.')
    else:
        print(f'Directory {directory_path} does not exist.')


def create_directories(input_path):
    if not os.path.isdir(input_path):
        print(f"The provided path '{input_path}' is not a valid directory.")
        return

    images_path = os.path.join(input_path, 'images')
    poses_path = os.path.join(input_path, 'poses')
    renders_path = os.path.join(input_path, 'renders')

    if not os.path.exists(images_path) or not os.path.exists(poses_path) or not os.path.exists(renders_path):
        print(f"The provided path '{input_path}' must contain both 'images' and 'poses' directories.")
        return

    new_directory_path = input_path.rstrip('/') + '_pent'
    new_images_path = os.path.join(new_directory_path, 'images')
    new_poses_path = os.path.join(new_directory_path, 'poses')
    new_renders_path = os.path.join(new_directory_path, 'renders')

    os.makedirs(new_images_path, exist_ok=True)
    os.makedirs(new_poses_path, exist_ok=True)
    os.makedirs(new_renders_path, exist_ok=True)
    os.makedirs(f"{new_directory_path}/sparse/0", exist_ok=True)

    images_files = sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0]))
    poses_files = sorted(os.listdir(poses_path), key=lambda x: int(x.split('.')[0]))

    if len(images_files) != len(poses_files):
        print("Number of images and poses do not match!")
        return

    for i in range(0, len(images_files), 5):
        new_index = i // 5
        new_image_filename = f"{new_index}.png"
        new_pose_filename = f"{new_index + 1}.txt"

        src_image_path = os.path.join(images_path, images_files[i])
        dst_image_path = os.path.join(new_images_path, new_image_filename)
        shutil.copyfile(src_image_path, dst_image_path)

        src_pose_path = os.path.join(poses_path, poses_files[i])
        dst_pose_path = os.path.join(new_poses_path, new_pose_filename)
        shutil.copyfile(src_pose_path, dst_pose_path)

        for ext in ['jpg', 'npy', 'png']:
            src_render_path = os.path.join(renders_path, f"{i}.{ext}")
            dst_render_filename = f"{new_index}.{ext}"
            dst_render_path = os.path.join(new_renders_path, dst_render_filename)
            if os.path.exists(src_render_path):
                shutil.copyfile(src_render_path, dst_render_path)

    print(f"Processed '{input_path}' successfully!")
    return new_directory_path


def run_commands(scene_path):
    print(f"Running recon.py with base directory '{scene_path}' and poses directory '{scene_path}/poses/'")
    subprocess.run([
        "python", "recon.py",
        f"{scene_path}/",
        f"{scene_path}/poses/",
        "rec.ply",
        f"{scene_path}/sparse/0/",
        "30"
    ])
    print("Completed running recon.py")


def process_scene(scene_path):
    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]

    path_parts = scene_path.split('/')
    scene_name = path_parts[-2]
    room_name = path_parts[-1]
    room_name_oct = room_name + '_pent'

    print(f"Processing scene '{scene_name}/{room_name_oct}'")

    scene_path_half = create_directories(scene_path)
    if scene_path_half:
        run_commands(scene_path_half)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process one or multiple scene directories.")
    parser.add_argument("scene_path", help="Path to the scene directory or a .txt file with scene paths")
    parser.add_argument("--front_path", required=True, help="Path to the 3D-FRONT directory")
    parser.add_argument("--future_path", required=True, help="Path to the 3D-FUTURE directory")
    parser.add_argument("--resources_path", required=True, help="Path to the resources directory")
    parser.add_argument("--scenes_path", required=True, help="Path to the base scenes directory")

    args = parser.parse_args()

    # Check if the provided scene_path is a directory or a .txt file
    if os.path.isdir(args.scene_path):
        process_scene(args.scene_path)
    elif os.path.isfile(args.scene_path) and args.scene_path.endswith('.txt'):
        with open(args.scene_path, 'r') as file:
            for line in file:
                scene_path = line.strip()
                scene_path = os.path.join(args.scenes_path, scene_path)
                if os.path.isdir(scene_path):
                    process_scene(scene_path)
                else:
                    print(f"Scene path '{scene_path}' in the .txt file does not exist or is not a valid directory.")
    else:
        print(f"The provided scene_path '{args.scene_path}' is not a valid directory or .txt file.")
        sys.exit(1)

    print("Script completed successfully")
