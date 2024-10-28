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


def extract_names(config_path):
    base_name = os.path.basename(config_path)
    scene_name, room_name_with_ext = base_name.split('_', 1)
    room_name = room_name_with_ext.replace('.json', '')
    print(f"Extracted scene name: {scene_name}, room name: {room_name}")
    return scene_name, room_name


def create_directories(scene_name, room_name, scenes_path):
    base_dir = os.path.join(scenes_path, scene_name, room_name)
    delete_directory(base_dir)
    os.makedirs(os.path.join(base_dir, "poses"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "sparse/0"), exist_ok=True)
    print(f"Created directories for scene '{scene_name}' and room '{room_name}'")


def run_commands(scene_name, room_name, config_path, visualize, front_path, future_path, resources_path, scenes_path):
    base_dir = os.path.join(scenes_path, scene_name, room_name)

    # Run spiral.py
    if visualize:
        print(
            f"Running spiral.py with output directory '{base_dir}/poses/' and config '{config_path}' with visualization")
        subprocess.run([
            "python", "spiral.py",
            "--output", os.path.join(base_dir, "poses"),
            "--config", config_path, "--visualize"
        ])
    else:
        print(f"Running spiral.py with output directory '{base_dir}/poses/' and config '{config_path}'")
        subprocess.run([
            "python", "spiral.py",
            "--output", os.path.join(base_dir, "poses"),
            "--config", config_path
        ])
    print("Completed running spiral.py")

    # Run blenderproc
    print(f"Running blenderproc with scene '{scene_name}'")
    subprocess.run([
        "blenderproc", "run", "examples/datasets/front_3d_with_improved_mat/main_short.py",
        os.path.join(front_path, f"{scene_name}.json"),
        future_path,
        future_path,
        resources_path,
        os.path.join(base_dir, "renders"),
        os.path.join(base_dir, "poses")
    ])
    print("Completed running blenderproc")

    # Run recon.py
    print(f"Running recon.py with base directory '{base_dir}' and poses directory '{base_dir}/poses/'")
    subprocess.run([
        "python", "recon.py",
        base_dir,
        os.path.join(base_dir, "poses"),
        "rec.ply",
        os.path.join(base_dir, "sparse/0"),
        "30"
    ])
    print("Completed running recon.py")


def process_config_file(config_path, visualize, front_path, future_path, resources_path, scenes_path):
    # Check if config_path is a JSON file or a TXT file
    if config_path.endswith('.json'):
        scene_name, room_name = extract_names(config_path)
        create_directories(scene_name, room_name, scenes_path)
        run_commands(scene_name, room_name, config_path, visualize, front_path, future_path, resources_path, scenes_path)
    elif config_path.endswith('.txt'):
        with open(config_path, 'r') as file:
            for line in file:
                config_file_path = line.strip()
                if os.path.isfile(config_file_path) and config_file_path.endswith('.json'):
                    scene_name, room_name = extract_names(config_file_path)
                    create_directories(scene_name, room_name, scenes_path)
                    run_commands(scene_name, room_name, config_file_path, visualize, front_path, future_path, resources_path, scenes_path)
                else:
                    print(f"Config file '{config_file_path}' does not exist or is not a JSON file.")
    else:
        print(f"Config file '{config_path}' is not a valid JSON or TXT file.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process a configuration file or a list of configuration files.")
    parser.add_argument("config_path", help="Path to the configuration JSON or TXT file")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--front_path", required=True, help="Path to the 3D-FRONT directory")
    parser.add_argument("--future_path", required=True, help="Path to the 3D-FUTURE directory")
    parser.add_argument("--resources_path", required=True, help="Path to the resources directory")
    parser.add_argument("--scenes_path", required=True, help="Path to the base scenes directory")

    args = parser.parse_args()

    config_path = args.config_path
    visualize = args.visualize
    front_path = args.front_path
    future_path = args.future_path
    resources_path = args.resources_path
    scenes_path = args.scenes_path

    if not os.path.isfile(config_path):
        print(f"Config file '{config_path}' does not exist.")
        sys.exit(1)

    print(f"Starting script with config file: {config_path}")
    process_config_file(config_path, visualize, front_path, future_path, resources_path, scenes_path)
    print("Script completed successfully")
