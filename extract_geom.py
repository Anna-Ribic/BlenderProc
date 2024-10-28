import os
import subprocess
import argparse

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run BlenderProc on selected JSON files.")
    parser.add_argument(
        "--json_list_file",
        type=str,
        default="../front/3D-FRONT/json_files_list.txt",
        help="Path to the file listing JSON files to process (default: '../front/3D-FRONT/json_files_list.txt')."
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="../front/3D-FRONT",
        help="Directory containing JSON files (default: '../front/3D-FRONT')."
    )
    parser.add_argument(
        "--future_model_dir",
        type=str,
        default="../front/3D-FUTURE-model",
        help="Directory containing 3D-FUTURE-model files (default: '../front/3D-FUTURE-model')."
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        default="resources",
        help="Directory containing resources (default: 'resources')."
    )
    parser.add_argument(
        "--blenderproc_script",
        type=str,
        default="examples/datasets/front_3d_with_improved_mat/geom.py",
        help="Path to the BlenderProc script (default: 'examples/datasets/front_3d_with_improved_mat/geom.py')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scenes/",
        help="Path to output directory (default: 'scenes/')."
    )
    return parser.parse_args()

# Function to run the BlenderProc command
def run_blenderproc(json_file, json_dir, blenderproc_script, future_model_dir, resources_dir, output_dir):
    json_path = os.path.join(json_dir, json_file)
    command = [
        "blenderproc",
        "run",
        blenderproc_script,
        json_path,
        future_model_dir,
        future_model_dir,  # Assuming the same directory is repeated intentionally
        resources_dir,
        output_dir,
    ]
    subprocess.run(command)

def main():
    args = parse_arguments()

    # Read JSON files from the provided file
    with open(args.json_list_file, "r") as f:
        json_files = [line.strip() for line in f if line.strip()]

    # Process each JSON file in the list
    for json_file in json_files:
        print(f"Processing {json_file}")
        run_blenderproc(json_file, args.json_dir, args.blenderproc_script, args.future_model_dir, args.resources_dir, args.output_dir)

if __name__ == "__main__":
    main()
