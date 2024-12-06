import os
import numpy as np
import open3d as o3d
import argparse
import json

def generate_spiral(a, b, theta_max, num_samples, x0, y0, z0, z_step, downward_angle_range):
    thetas = np.linspace(0, theta_max, num_samples)
    rs = a + b * thetas

    xs = x0 + rs * np.cos(thetas)
    ys = y0 + rs * np.sin(thetas)
    zs = z0 + (thetas / (2 * np.pi)) * z_step

    positions = np.vstack((xs, ys, zs)).T
    directions_outward = np.vstack((np.cos(thetas), np.sin(thetas), np.zeros_like(thetas))).T
    directions_outward = directions_outward / np.linalg.norm(directions_outward, axis=1, keepdims=True)

    return positions, directions_outward

def add_frustum(mesh, position, direction, size=0.1, fov=np.pi / 4):
    aspect_ratio = 1.0
    near_clip = 0.1 * size
    far_clip = size

    h_near = 2 * np.tan(fov / 2) * near_clip
    w_near = h_near * aspect_ratio
    h_far = 2 * np.tan(fov / 2) * far_clip
    w_far = h_far * aspect_ratio

    forward = direction
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)

    near_center = position + forward * near_clip
    far_center = position + forward * far_clip

    near_top_left = near_center + (up * h_near / 2) - (right * w_near / 2)
    near_top_right = near_center + (up * h_near / 2) + (right * w_near / 2)
    near_bottom_left = near_center - (up * h_near / 2) - (right * w_near / 2)
    near_bottom_right = near_center - (up * h_near / 2) + (right * w_near / 2)

    far_top_left = far_center + (up * h_far / 2) - (right * w_far / 2)
    far_top_right = far_center + (up * h_far / 2) + (right * w_far / 2)
    far_bottom_left = far_center - (up * h_far / 2) - (right * w_far / 2)
    far_bottom_right = far_center - (up * h_far / 2) + (right * w_far / 2)

    vertices = [
        near_top_left, near_top_right, near_bottom_right, near_bottom_left,
        far_top_left, far_top_right, far_bottom_right, far_bottom_left
    ]

    num_vertices = len(mesh.vertices)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack((mesh_vertices, vertices)))

    triangles = [
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 4, 1], [1, 4, 5],
        [1, 5, 2], [2, 5, 6],
        [2, 6, 3], [3, 6, 7],
        [3, 7, 0], [0, 7, 4]
    ]

    triangles = [[idx + num_vertices for idx in tri] for tri in triangles]
    mesh_triangles = np.asarray(mesh.triangles)
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack((mesh_triangles, triangles)))

def rotate_direction_downward(direction, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    forward = direction
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    rotation_matrix = np.array([
        [cos_angle + right[0] * right[0] * (1 - cos_angle),
         right[0] * right[1] * (1 - cos_angle) - right[2] * sin_angle,
         right[0] * right[2] * (1 - cos_angle) + right[1] * sin_angle],

        [right[1] * right[0] * (1 - cos_angle) + right[2] * sin_angle,
         cos_angle + right[1] * right[1] * (1 - cos_angle),
         right[1] * right[2] * (1 - cos_angle) - right[0] * sin_angle],

        [right[2] * right[0] * (1 - cos_angle) - right[1] * sin_angle,
         right[2] * right[1] * (1 - cos_angle) + right[0] * sin_angle,
         cos_angle + right[2] * right[2] * (1 - cos_angle)]
    ])

    downward_direction = np.dot(rotation_matrix, forward)
    return downward_direction

def compute_transformation_matrix(position, direction):
    forward = direction / np.linalg.norm(direction)
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation_matrix = np.vstack((right, up, -forward)).T
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix

def generate_heatmap_gradient(num_samples):
    """Generate a heatmap gradient from red to orange, yellow, green, and blue."""
    # Define the colors for the gradient
    colors = np.array([
        [1, 0.3, 0],    # Red
        [1, 0.5, 0],  # Orange
        [1, 1, 0],    # Yellow
        [0, 1, 0],    # Green
        [0, 1, 1],  # turqoise?
        [0, 0, 1]     # Blue
    ])

    # Define interpolation points
    num_colors = len(colors)
    color_indices = np.linspace(0, num_colors - 1, num_samples)

    # Interpolate the colors
    interpolated_colors = np.empty((num_samples, 3))
    for i in range(3):  # For R, G, B channels
        interpolated_colors[:, i] = np.interp(color_indices, np.arange(num_colors), colors[:, i])

    return interpolated_colors


def get_lower_80_percent(mesh):
    # Convert mesh vertices to numpy array
    vertices = np.asarray(mesh.vertices)

    # Find the height range (z-axis values)
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])

    # Calculate the threshold for the lower 80% of height
    z_threshold = min_z + 0.8 * (max_z - min_z)

    # Filter vertices to keep only those below the threshold
    mask = vertices[:, 2] <= z_threshold
    filtered_indices = np.where(mask)[0]
    filtered_vertices = vertices[mask]

    # Create a mapping from old vertex indices to new indices
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(filtered_indices)}

    # Find the faces that are connected to the remaining vertices
    faces = np.asarray(mesh.triangles)
    filtered_faces = []
    for face in faces:
        if all(vertex in index_mapping for vertex in face):
            new_face = [index_mapping[vertex] for vertex in face]
            filtered_faces.append(new_face)

    # Create a new mesh with the filtered vertices and faces
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(filtered_faces)

    # Estimate normals for the new mesh
    new_mesh.compute_vertex_normals()
    return new_mesh


def main(config_file, output_dir, visualize):
    with open(config_file, 'r') as file:
        config = json.load(file)

    spiral_params_outside = config.get("spiral_params_outside", [])
    spiral_params_inside = config.get("spiral_params_inside", [])
    unique_id = os.path.basename(config_file).split('_')[0]
    """spiral_params_outside = [{'a': 0.0005, 'b': 0.03, 'theta_max': 3 * np.pi, 'num_samples': 20, 'x0': -4.15, 'y0': -2.5, 'z0': 1.1, 'z_step': 0.3, 'downward_angle_range': (10, 45)},
    {'a': 0.0005, 'b': 0.04, 'theta_max': 4 * np.pi, 'num_samples': 20, 'x0': -2.1, 'y0': -2, 'z0': 1.2, 'z_step': 0.3, 'downward_angle_range': (10, 50)},
                             {'a': 0.0005, 'b': 0.04, 'theta_max': 4 * np.pi, 'num_samples': 20, 'x0': -1, 'y0': -2,
                              'z0': 0.85, 'z_step': 0.3, 'downward_angle_range': (15, 30)},
    ]

    spiral_params_inside = [
       {'a': 0.8, 'b': 0.05, 'theta_max': 5 * np.pi, 'num_samples': 60, 'x0': -2.1, 'y0': -2, 'z0': 0.95, 'z_step': 0.3,
        'downward_angle_range': (10, 55)},
    ]"""

    mesh = o3d.geometry.TriangleMesh()
    combined_geometry = []
    pose_count = 1

    os.makedirs(output_dir, exist_ok=True)

    start_color = np.array([1, 0.5, 0])  # Red
    end_color = np.array([0, 0, 1])  # Turquoise

    for params in spiral_params_outside:
        positions, directions_outward = generate_spiral(**params)
        directions_outward = -directions_outward

        downward_angles = np.linspace(params['downward_angle_range'][0], params['downward_angle_range'][1], params['num_samples'])
        directions_outward = np.array([rotate_direction_downward(dir, angle) for dir, angle in zip(directions_outward, downward_angles)])

        colors = generate_heatmap_gradient(params['num_samples'])


        spiral_pcd = o3d.geometry.PointCloud()
        spiral_pcd.points = o3d.utility.Vector3dVector(positions)
        spiral_pcd.colors = o3d.utility.Vector3dVector(colors)  # Apply gradient to point cloud
        combined_geometry.append(spiral_pcd)

        for pos, dir in zip(positions, directions_outward):
            add_frustum(mesh, pos, dir)

        for pos, dir in zip(positions, directions_outward):
            transformation_matrix = compute_transformation_matrix(pos, dir)
            file_path = os.path.join(output_dir, f"{pose_count}.txt")
            np.savetxt(file_path, transformation_matrix, fmt='%.6f')
            pose_count += 1
            print(f"Saved outward camera pose {pose_count - 1} to {file_path}")

    for params in spiral_params_inside:
        positions, directions_inward = generate_spiral(**params)

        downward_angles = np.linspace(params['downward_angle_range'][0], params['downward_angle_range'][1], params['num_samples'])
        directions_inward = np.array([rotate_direction_downward(dir, angle) for dir, angle in zip(directions_inward, downward_angles)])

        colors = generate_heatmap_gradient(params['num_samples'])


        spiral_pcd = o3d.geometry.PointCloud()
        spiral_pcd.points = o3d.utility.Vector3dVector(positions)
        spiral_pcd.colors = o3d.utility.Vector3dVector(colors)  # Apply gradient to point cloud
        combined_geometry.append(spiral_pcd)

        for pos, dir in zip(positions, directions_inward):
            add_frustum(mesh, pos, dir)

        for pos, dir in zip(positions, directions_inward):
            transformation_matrix = compute_transformation_matrix(pos, dir)
            file_path = os.path.join(output_dir, f"{pose_count}.txt")
            np.savetxt(file_path, transformation_matrix, fmt='%.6f')
            pose_count += 1
            print(f"Saved inward camera pose {pose_count - 1} to {file_path}")

    mesh.paint_uniform_color([1, 1, 1])
    combined_geometry.append(mesh)

    if visualize:

        existing_mesh = o3d.io.read_triangle_mesh(os.path.join('scenes', unique_id, 'geom.ply'))
        existing_mesh = get_lower_80_percent(existing_mesh)
        combined_geometry.append(existing_mesh)

        o3d.visualization.draw_geometries(combined_geometry, point_show_normal=True, mesh_show_wireframe=True)
    #o3d.io.write_triangle_mesh("spiral_with_frustums.ply", )
    #print("Exported the mesh to spiral_with_frustums.ply")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate spiral camera poses and frustums.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the output files.')
    parser.add_argument('--visualize', action='store_true', help='Visualize Camera Poses.')
    args = parser.parse_args()

    main(args.config, args.output, args.visualize)
