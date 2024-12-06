import json
import numpy as np
import argparse
import os
import random
import math


def is_point_inside_bounding_box(point, bounding_box, margin=0.3):
    """
    Check if a point is inside a bounding box with a margin of error.

    Parameters:
    point (tuple): The coordinates of the point as (x, y, z).
    bounding_box (tuple): The coordinates of the bounding box as (min_x, min_y, min_z, max_x, max_y, max_z).
    margin (float): The margin of error to include when checking if the point is inside the bounding box.

    Returns:
    bool: True if the point is inside the bounding box with the margin of error, False otherwise.
    """
    x, z, y = point
    min_x, min_z, min_y, max_x, max_z, max_y = bounding_box

    if (min_x - margin <= x <= max_x + margin and
        min_y - margin <= y <= max_y + margin):
        return True
    return False

def generate_spiral_params_outside(child_pos,a=0.0005, b=0.01, angle_range=(15, 40), num_samples=20, theta=12.42):
    return {
        "a": a,
        "b": b,
        "theta_max": theta,
        "num_samples": num_samples,
        "x0": child_pos[0],
        "y0": child_pos[2],
        "z0": child_pos[1],
        "z_step": 0.2,
        "downward_angle_range": [angle_range[0], angle_range[1]],
    }

def generate_spiral_params_inside(child_pos, a, num_samples=50, theta=11.7):
    return {
        "a": a,
        "b": 0.05,
        "theta_max": theta,
        "num_samples": num_samples,
        "x0": child_pos[0],
        "y0": child_pos[2],
        "z0": child_pos[1],
        "z_step": 0.3,
        "downward_angle_range": [15, 60]
    }

def calculate_room_bounding_boxes(data):
    furniture = data['furniture']
    data = data['scene']

    room_spiral_params = []

    rooms = data.get('room', [])
    for room_data in rooms:
        room_name = room_data.get('type', 'Unnamed Room').replace(" ", "_")
        outside_spirals = []
        inside_spirals = []

        children = room_data.get('children', [])

        min_x = float('inf')
        min_y = float('inf')
        min_z = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        max_z = float('-inf')
        print('\n')
        print(room_name)

        covered_room = []
        spiral_parmams_list=[]

        covered_room_low = []

        for child in children:
            #print(child['ref'])
            child_pos = child.get('pos', [0, 0, 0])
            child_scale = child.get('scale', [1, 1, 1])
            if child_pos == [0, 0, 0]:
                continue

            print(child['ref'])
            bbox = None
            for f in furniture:
                if child['ref']== f['uid']:
                    bbox = f.get('bbox', None)
                    print('category', f.get('category', None))

            if bbox is not None:
                print('bbox', bbox)
                if len(bbox) == 1:
                    bbox = bbox[0]
                child_scale[0] *= bbox[0]
                child_scale[1] *= bbox[1]
                child_scale[2] *= bbox[2]

            half_x = child_scale[0] / 2
            half_y = child_scale[2] / 2
            half_z = child_scale[1] / 2

            child_min_x = child_pos[0] - half_x
            child_min_y = child_pos[2] - half_y
            child_min_z = child_pos[1] - half_z
            child_max_x = child_pos[0] + half_x
            child_max_y = child_pos[2] + half_y
            child_max_z = child_pos[1] + half_z

            if bbox is not None:
                covered_room_low.append((child_min_x, child_min_z, child_min_y, child_max_x, child_max_z, child_max_y))

            min_x = min(min_x, child_min_x)
            min_y = min(min_y, child_min_y)
            min_z = 0
            max_x = max(max_x, child_max_x)
            max_y = max(max_y, child_max_y)
            max_z = 2.5

            if child_max_z > 1.1 or bbox is None:
                print('max_z', child['ref'], child_max_z)
                if bbox is None:
                    continue
                else:
                    print('Add to covered room')
                    covered_room.append((child_min_x, child_min_z, child_min_y, child_max_x, child_max_z, child_max_y))
                    continue

            spiral_params_outside = generate_spiral_params_outside([child_pos[0], min(child_max_z + 0.3, 1.3), child_pos[2]])
            outside_spirals.append(spiral_params_outside)

            """print('Spiral params', child_pos[0], min(child_max_z + 0.3, 1.3), child_pos[2])
            spiral_parmams_list.append([child_pos[0], min(child_max_z + 0.3, 1.3), child_pos[2]])

        for s in spiral_parmams_list:
            check = any([is_point_inside_bounding_box(s, r) for r in covered_room])
            if not check:
                spiral_params_outside = generate_spiral_params_outside(s)
                outside_spirals.append(spiral_params_outside)
            else:
                print('Spiral in bounding box')"""
        check_nan = any([not math.isfinite(x) for x in [min_x, min_y, min_z, max_x, max_y, max_z]])
        if check_nan:
            continue


        middle_x = (min_x + max_x) / 2
        middle_y = (min_y + max_y) / 2
        middle_z = 0.9

        print([min_x, min_y, min_z], [max_x, max_y, max_z])

        spiral_params_inside = generate_spiral_params_inside([middle_x, middle_z, middle_y], 0.2 * min(np.abs(min_x - max_x), np.abs(min_y - max_y)), num_samples=80, theta=5*np.pi)
        inside_spirals.append(spiral_params_inside)

        spiral_up_middle = generate_spiral_params_outside([middle_x, middle_z, middle_y], a=0.05 * min(np.abs(min_x - max_x), np.abs(min_y - max_y)),b=0.01, angle_range=[-10, -35], num_samples=20, theta=5*np.pi)
        outside_spirals.append(spiral_up_middle)

        spiral_down_middle = generate_spiral_params_outside([middle_x, 1.2, middle_y],
                                                          a=0.05 * min(np.abs(min_x - max_x), np.abs(min_y - max_y)),
                                                          b=0.01, angle_range=[10, 55], num_samples=40, theta=5 * np.pi)
        outside_spirals.append(spiral_down_middle)

        width = max_x - min_x
        height = max_y - min_y
        depth = max_z - min_z

        small_width = 0.4 * width
        small_height = 0.4 * height
        small_depth = 0.4 * depth

        offset_x = (width - small_width) / 2
        offset_y = (height - small_height) / 2
        offset_z = (depth - small_depth) / 2

        small_min_x = min_x + offset_x
        small_max_x = max_x - offset_x
        small_min_y = min_y + offset_y
        small_max_y = max_y - offset_y
        small_min_z = min_z + offset_z
        small_max_z = max_z - offset_z

        corner1 = (small_min_x, 1.1, small_min_y)
        corner2 = (small_max_x, 1.1, small_min_y)
        corner3 = (small_min_x, 1.1, small_max_y)
        corner4 = (small_max_x, 1.1, small_max_y)

        corners = [corner1, corner2, corner3, corner4]
        print('covered_room', covered_room)
        for corner in corners:
            print('corner', corner)
            check = any([is_point_inside_bounding_box(corner, r) for r in covered_room])
            if check:
                print('Corner in bounding box\n')
                continue
            spiral_params_outside = generate_spiral_params_outside(corner, b=0.01, angle_range=[0, -15])
            outside_spirals.append(spiral_params_outside)

        random_count = 0
        tries= 0
        while random_count< 3 and tries < 100:
            x = random.uniform(small_min_x, small_max_x)
            y = random.uniform(small_min_y, small_max_y)
            point = (x, 0.2, y )
            check = any([is_point_inside_bounding_box(point, r, margin=0.1) for r in covered_room_low])
            if not check:
                random_spiral = generate_spiral_params_outside(point, b=0.01, angle_range=[-5, -30], num_samples=20,)
                covered_room_low.append((point[0]-0.4,point[1]-0.1,point[2]-0.1, point[0]+0.1, point[1]+0.1, point[2]+0.1))
                outside_spirals.append(random_spiral)
                random_count += 1
            tries += 1

        room_params = {
            'room_name': room_name,
            'spiral_params_outside': outside_spirals,
            'spiral_params_inside': inside_spirals
        }

        room_spiral_params.append(room_params)

    return room_spiral_params

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate spiral parameters from JSON file(s) located at a base path.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to a JSON file or a text file containing JSON file paths.")
    parser.add_argument("--global_txt_file", type=str, required=True, help="Path to the global text file where output paths will be appended.")
    parser.add_argument("--output_dir", type=str, default="configs", help="Directory to store generated config files.")
    parser.add_argument("--base_path", type=str, default="", help="Base path to prepend to each JSON file path.")
    return parser.parse_args()

def load_json_paths(input_path, base_path=""):
    """
    Load JSON paths from the input file.
    If the file is a .txt, read each line as a JSON path and prepend base_path.
    If the file is a .json, return the path with base_path prepended as a single-item list.
    """
    if input_path.endswith('.txt'):
        with open(input_path, 'r') as file:
            json_paths = [os.path.join(base_path, line.strip()) for line in file if line.strip()]
    elif input_path.endswith('.json'):
        json_paths = [os.path.join(base_path, input_path)]
    else:
        raise ValueError("Input file must be a JSON or a TXT file containing JSON paths.")
    return json_paths

def process_json_file(json_file_path, global_txt_file, output_dir):
    # Load JSON data from file
    print(json_file_path)
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Calculate room spiral parameters
    room_spiral_params = calculate_room_bounding_boxes(data)
    base_filename = os.path.splitext(os.path.basename(json_file_path))[0]

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    with open(global_txt_file, 'a') as txt_file:
        for room in room_spiral_params:
            if len(room['spiral_params_outside']) < 10:
                continue
            room_name = room['room_name']
            dic = {
                'spiral_params_outside': room['spiral_params_outside'],
                'spiral_params_inside': room['spiral_params_inside']
            }

            # Save outside spirals
            outside_filename = os.path.join(output_dir, f"{base_filename}_{room_name}.json")
            with open(outside_filename, 'w') as file:
                json.dump(dic, file, indent=2)
            print(f"Saved {outside_filename}")

            # Append the created file path to the global text file
            txt_file.write(outside_filename + '\n')

def main():
    args = parse_arguments()
    input_path = args.input_file
    global_txt_file = args.global_txt_file
    output_dir = args.output_dir
    base_path = args.base_path

    # Get list of JSON paths with base path prepended
    json_paths = load_json_paths(input_path, base_path)

    # Process each JSON file
    for json_file_path in json_paths:
        process_json_file(json_file_path, global_txt_file, output_dir)

if __name__ == "__main__":
    main()
