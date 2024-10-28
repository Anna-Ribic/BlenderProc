import os
import sys
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import re

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def load_images_depths(image_depth_dir):
    images = []
    depths = []
    num_images = 0

    for filename in sorted(os.listdir(image_depth_dir), key=natural_sort_key):
        if filename.endswith('.png'):
            img_path = os.path.join(image_depth_dir, filename)
            depth_path = os.path.join(image_depth_dir, f"{os.path.splitext(filename)[0]}.npy")  # Assuming depth maps have the same name with .npy extension
            img = np.array(Image.open(img_path))  # Load PNG image
            depth = np.load(depth_path)  # Load depth map
            images.append(img)
            depths.append(depth)
            num_images += 1

    return images, depths, num_images

def load_poses(pose_dir, num_images):
    poses = []
    for i in range(1, num_images + 1):  # Assuming 1.txt corresponds to image 0.png, 2.txt to 1.png, etc.
        pose = np.loadtxt(os.path.join(pose_dir, f'{i}.txt'))

        R = pose[:3, :3]
        t = pose[:3, 3]

        # Rotation matrix for 180 degrees around the local Z-axis (viewing direction)
        R_z_180 = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

        # Apply the rotation to the camera's rotation matrix
        new_R = R @ R_z_180

        # Construct the new 4x4 pose matrix
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_R
        new_pose[:3, 3] = t

        poses.append(new_pose)

    return poses

def save_ply_cloud(points, filename):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for point, color in points:
            f.write('{} {} {} {} {} {}\n'.format(point[0], point[1], point[2], color[0], color[1], color[2]))

    print(f"Point cloud PLY file saved: {filename}")

def create_camera_pyramid(matrix, pyramid_size=0.1):
    origin = matrix[:3, 3]
    R = matrix[:3, :3]

    # Define the pyramid base and tip
    base = pyramid_size * np.array([
        [1,  1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1,  1, 1]
    ])
    tip = np.array([0, 0, 0])

    # Apply the rotation and translation
    base_transformed = (R @ base.T).T + origin
    tip_transformed = R @ tip + origin

    # Create vertices
    vertices = np.vstack((base_transformed, tip_transformed))
    return vertices

def save_ply_cameras(cameras, filename):
    vertices = []
    vertex_idx = 0
    faces = []

    for cam in cameras:
        pose = cam['pose']
        pyramid = create_camera_pyramid(pose)

        # Add the pyramid vertices to the list
        vertices.extend(pyramid)

        # Create the faces (4 faces for the pyramid)
        tip_index = vertex_idx + 4
        for i in range(4):
            next_i = (i + 1) % 4
            faces.append([vertex_idx + i, vertex_idx + next_i, tip_index])
        vertex_idx += 5

    # Create PLY elements
    vertices_np = np.array(vertices)
    vertex_elements = [(vertex[0], vertex[1], vertex[2]) for vertex in vertices_np]
    face_elements = [(face,) for face in faces]

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    face_dtype = [('vertex_indices', 'i4', (3,))]

    vertex_element = PlyElement.describe(np.array(vertex_elements, dtype=vertex_dtype), 'vertex')
    face_element = PlyElement.describe(np.array(face_elements, dtype=face_dtype), 'face')

    ply_data = PlyData([vertex_element, face_element], text=True)
    ply_data.write(filename)

    print(f"Cameras PLY file saved: {filename}")

def visualize_depth_image(depth_image):
    # Visualize depth map
    plt.figure()
    plt.imshow(depth_image, cmap='gray')
    plt.title('Depth Map')
    plt.colorbar()
    plt.axis('off')
    plt.show()

def invert_pose(rotation_matrix, translation_vector):
    # Invert the rotation matrix
    inverted_rotation_matrix = np.transpose(rotation_matrix)

    # Invert the translation vector and apply the inverted rotation
    inverted_translation_vector = - inverted_rotation_matrix @ translation_vector

    return inverted_rotation_matrix, inverted_translation_vector


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def normalize_rotation_matrix(R):
    # Ensure R is a proper rotation matrix
    U, _, Vt = np.linalg.svd(R)
    return np.dot(U, Vt)

def matrix_to_quaternion_translation(matrix):
    # Extract rotation matrix
    global b
    rotation_matrix = matrix[:3, :3]
    #r = R.from_matrix(rotation_matrix)

    # Extract the valid rotation matrix
    #rotation_matrix = r.as_matrix()

    # Extract translation vector
    translation_vector = matrix[:3, 3]

    matrix[:3,:3] = rotation_matrix


    #print('trans1', translation_vector)

    # Invert pose
    inverted_rotation_matrix, inverted_translation_vector = invert_pose(rotation_matrix, translation_vector)

    """#print('inerted trans', inverted_translation_vector)
    #print(- rotation_matrix @ - rotation_matrix.T)
    #print(- rotation_matrix @ - rotation_matrix.T @ translation_vector)

    #print('back', - inverted_rotation_matrix.T @ inverted_translation_vector - translation_vector)

    #print('forward', inverted_rotation_matrix)

    # Convert rotation matrix to quaternion
    quaternion = rotmat2qvec(inverted_rotation_matrix)
    #print('test',np.allclose( inverted_rotation_matrix.T @ inverted_rotation_matrix, np.eye(3)))
    #print('val',inverted_rotation_matrix.T @ inverted_rotation_matrix )
    print('quat', np.allclose(qvec2rotmat(quaternion) ,inverted_rotation_matrix, rtol=1e-01, atol=1))
    print('quat', qvec2rotmat(quaternion), inverted_rotation_matrix)"""

    inverted_rotation_matrix = normalize_rotation_matrix(inverted_rotation_matrix)

    are_close = np.allclose(np.eye(3), inverted_rotation_matrix.T @ inverted_rotation_matrix, rtol=1e-02, atol=1e-04)
    print('Is the rot matrix valid? ', are_close)
    if not are_close:
        print(inverted_rotation_matrix.T @ inverted_rotation_matrix)

    # Convert rotation matrix to quaternion
    quaternion = rotmat2qvec(inverted_rotation_matrix)

    # Convert quaternion back to rotation matrix
    rotmat_from_quat = qvec2rotmat(quaternion)

    print('Determinant', print(np.linalg.det(rotation_matrix)))
    print('Determinant', print(np.linalg.det(inverted_rotation_matrix)))

    # Check if the rotation matrix is close to the original one
    are_close = np.allclose(rotmat_from_quat, inverted_rotation_matrix, rtol=1e-05, atol=1e-08)
    print('Are the original and converted rotation matrices close? ', are_close)
    if not are_close:
        print(inverted_rotation_matrix)
        print(rotmat_from_quat)
        print('Difference between matrices:', rotmat_from_quat - inverted_rotation_matrix)

    return quaternion, inverted_translation_vector

def save_colmap_format(point_cloud, output_dir, poses):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save cameras.txt
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write("1 PINHOLE 369 369 355.5 355.5 184.5 184.5\n")  # Example values

    # Save images.txt
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        for img_index, matrix in enumerate(poses):
            print(matrix, img_index)
            img_index+=1
            quaternion, translation = matrix_to_quaternion_translation(matrix)
            image_line = f"{img_index} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]} {translation[0]} {translation[1]} {translation[2]} 1 {img_index-1}.png\n"
            f.write(image_line)

            img_points = []
            for p, obs in point_cloud.items():
                id = obs['id']
                for i, u, v, _ in obs['observations']:
                    if i == img_index:
                        img_points.append(f"{u} {v} {id}")
            point_line = ' '.join(img_points) + '\n'
            f.write(point_line)

    # Save points3D.txt
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        for point, data in point_cloud.items():
            x, y, z = point
            point_id = data['id']
            r, g, b = data['color']
            error = 0.0  # Assuming 0.0 for the error
            observations = data['observations']
            track = ' '.join(f'{obs[0]} {obs[3]}' for obs in observations)
            #print('track', track)
            f.write(f"{point_id} {x} {y} {z} {r} {g} {b} {error} {track}\n")


import open3d as o3d



def visualize_ply(file_path):
    # Load the PLY file as a point cloud
    point_cloud = o3d.io.read_point_cloud(file_path)
    
    # Check if the file is successfully loaded
    if not point_cloud.is_empty():
        # Display the point cloud
        o3d.visualization.draw_geometries([point_cloud])
    else:
        print("Failed to load the PLY file.")


def display_image_with_marked_pixels(image: np.ndarray, pixels_to_mark: list, mark_color: tuple = (255, 0, 0)):
    """
    Displays an image with specified pixels marked.

    Parameters:
        image (np.ndarray): The input image as a numpy array.
        pixels_to_mark (list): A list of tuples, where each tuple represents the (row, col) of the pixel to mark.
        mark_color (tuple): The color to use for marking the pixels (default is red).
    """
    # Create a copy of the image to avoid modifying the original image
    marked_image = image.copy()

    # Mark the specified pixels
    for (row, col) in pixels_to_mark:
        marked_image[row, col] = mark_color

    # Display the image
    plt.imshow(marked_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def main(image_depth_dir, pose_dir, output_ply, sparse_dir, res=50):
    # Load images and depths from the same directory

    dir = image_depth_dir
    image_depth_dir = os.path.join(dir, 'renders')
    new_images_dir = os.path.join(dir, 'images')
    os.makedirs(new_images_dir, exist_ok=True)

    images, depths, num_images = load_images_depths(image_depth_dir)

    # Load camera poses
    poses = load_poses(pose_dir, num_images)

    print('nm',num_images)

    # Camera intrinsics (pinhole model)
    fx = 355.5
    fy = 355.5
    cx = 184.5
    cy = 184.5

    point_cloud = {}
    cameras = []
    point_id = 1

    for i in range(num_images):
        img = images[i]
        img = img[:,::-1]

        flipped_image = Image.fromarray(img)
        flipped_image.save(os.path.join(new_images_dir, str(i)+'.png'))

        depth = depths[i]
        depth = depth[:,::-1]
        pose = poses[i]

        height, width = img.shape[:2]
        step = max(height, width) // res  # Step size for 50x50 grid
        #print(step, height, width)
        pi = 0
        for v in range(0, height, step):
            for u in range(0, width, step):
                Z = depth[v, u]
                if Z == 0 or Z > 6:  # No depth available
                    continue

                pi +=1

                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy  # Inverting the Y axis

                # 3D point in camera coordinates
                P_camera = np.array([X, Y, Z, 1])

                # Transform to world coordinates
                P_world = np.dot(pose, P_camera)[:3]

                # Round to nearest 0.01 resolution in world space
                P_rounded = tuple(np.round(P_world / 0.0001) * 0.0001)

                # Color from RGB image
                color = img[v, u]

                # Store in dictionary with rounded position as key to avoid duplicates
                if P_rounded not in point_cloud:
                    point_cloud[P_rounded] = {
                        'id': point_id,
                        'color': color[:3],
                        'observations': [(i+1, u, v, pi)]
                    }
                    point_id += 1
                else:
                    #print('here')
                    point_cloud[P_rounded]['observations'].append((i+1, u, v, pi))
                    #print(point_cloud[P_rounded]['id'])

                #print(f'v: {v}, u: {u}')
                #print(img.shape)
                #display_image_with_marked_pixels(img[:,:,:3], [(v, u)])

        # Save camera pose with viewing direction
        cameras.append({'pose': pose})

    # Filter points that have been observed at least 5 times
    filtered_point_cloud = [(np.array(key), value['color']) for key, value in point_cloud.items() if len(value['observations']) >= 1]

    # Save point cloud to PLY with camera poses
    #save_ply_cloud(filtered_point_cloud, output_ply)

    #visualize_ply(output_ply)

    #save_ply_cameras(cameras, 'output_ply_cameras.ply')

    save_colmap_format(point_cloud, sparse_dir, poses)

    print(f"Point cloud saved to {output_ply}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py <image_depth_directory> <pose_directory> <output_ply_file> <output_sparse_dir> <res>")
        sys.exit(1)


    image_depth_dir = sys.argv[1]
    pose_dir = sys.argv[2]
    output_ply = sys.argv[3]
    output_sparse_dir = sys.argv[4]
    res = int(sys.argv[5])

    main(image_depth_dir, pose_dir, output_ply, output_sparse_dir, res)





"""import os
import sys
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import re


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def load_images_depths(image_depth_dir):
    images = []
    depths = []
    num_images = 0

    for filename in sorted(os.listdir(image_depth_dir), key=natural_sort_key):
        if filename.endswith('.png'):
            img_path = os.path.join(image_depth_dir, filename)
            depth_path = os.path.join(image_depth_dir, f"{os.path.splitext(filename)[0]}.npy")  # Assuming depth maps have the same name with .npy extension
            print(img_path, depth_path)
            img = np.array(Image.open(img_path))  # Load PNG image
            depth = np.load(depth_path)  # Load depth map
            images.append(img)
            depths.append(depth)
            num_images += 1

    return images, depths, num_images


def load_poses(pose_dir, num_images):
    poses = []
    for i in range(1, num_images + 1):  # Assuming 1.txt corresponds to image 0.png, 2.txt to 1.png, etc.
        pose = np.loadtxt(os.path.join(pose_dir, f'{i}.txt'))
        poses.append(pose)

    return poses


def save_ply_cloud(points, filename):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for point, color in points:
            f.write('{} {} {} {} {} {}\n'.format(point[0], point[1], point[2], color[0], color[1], color[2]))

    print(f"Point cloud PLY file saved: {filename}")


def create_camera_pyramid(matrix, pyramid_size=0.1):
    origin = matrix[:3, 3]
    R = matrix[:3, :3]

    # Define the pyramid base and tip
    base = pyramid_size * np.array([
        [1,  1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1,  1, 1]
    ])
    tip = np.array([0, 0, 0])

    # Apply the rotation and translation
    base_transformed = (R @ base.T).T + origin
    tip_transformed = R @ tip + origin

    # Create vertices
    vertices = np.vstack((base_transformed, tip_transformed))
    return vertices


def save_ply_cameras(cameras, filename):
    vertices = []
    vertex_idx = 0
    faces = []

    for cam in cameras:
        pose = cam['pose']
        pyramid = create_camera_pyramid(pose)

        # Add the pyramid vertices to the list
        vertices.extend(pyramid)

        # Create the faces (4 faces for the pyramid)
        tip_index = vertex_idx + 4
        for i in range(4):
            next_i = (i + 1) % 4
            faces.append([vertex_idx + i, vertex_idx + next_i, tip_index])
        vertex_idx += 5

    # Create PLY elements
    vertices_np = np.array(vertices)
    vertex_elements = [(vertex[0], vertex[1], vertex[2]) for vertex in vertices_np]
    face_elements = [(face,) for face in faces]

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    face_dtype = [('vertex_indices', 'i4', (3,))]

    vertex_element = PlyElement.describe(np.array(vertex_elements, dtype=vertex_dtype), 'vertex')
    face_element = PlyElement.describe(np.array(face_elements, dtype=face_dtype), 'face')

    ply_data = PlyData([vertex_element, face_element], text=True)
    ply_data.write(filename)

    print(f"Cameras PLY file saved: {filename}")


def visualize_depth_image(depth_image):
    # Visualize depth map
    plt.figure()
    plt.imshow(depth_image, cmap='gray')
    plt.title('Depth Map')
    plt.colorbar()
    plt.axis('off')
    plt.show()


def main(image_depth_dir, pose_dir, output_ply):
    # Load images and depths from the same directory
    images, depths, num_images = load_images_depths(image_depth_dir)

    # Load camera poses
    poses = load_poses(pose_dir, num_images)

    # Camera intrinsics (pinhole model)
    fx = 355.5
    fy = 355.5
    cx = 255.5
    cy = 255.5

    point_cloud = {}
    cameras = []

    for i in range(min(num_images, 100)):
        img = images[i]
        depth = depths[i]
        pose = poses[i]

        #visualize_depth_image(depth)

        height, width = img.shape[:2]
        step = max(height, width) // 50  # Step size for 50x50 grid
        for v in range(0, height, step):
            for u in range(0, width, step):
                Z = depth[v, u]
                if Z == 0 or Z > 6:  # No depth available
                    continue

                X = (u - cx) * Z / fx
                Y = (cy - v) * Z / fy  # Inverting the Y axis

                # 3D point in camera coordinates
                P_camera = np.array([X, Y, Z, 1])

                # Transform to world coordinates
                P_world = np.dot(pose, P_camera)[:3]

                # Round to nearest 0.01 resolution in world space
                P_rounded = tuple(np.round(P_world / 0.01) * 0.01)

                # Color from RGB image
                color = img[v, u]

                # Store in dictionary with rounded position as key to avoid duplicates
                if P_rounded not in point_cloud:
                    point_cloud[P_rounded] = {'color': color, 'count': 1}
                else:
                    point_cloud[P_rounded]['count'] += 1

        # Save camera pose with viewing direction
        cameras.append({'pose': pose})

    # Filter points that have been observed at least 5 times
    filtered_point_cloud = [(np.array(key), value['color']) for key, value in point_cloud.items() if value['count'] >= 1]

    # Save point cloud to PLY with camera poses
    save_ply_cloud(filtered_point_cloud, output_ply)

    save_ply_cameras(cameras, 'output_ply_cameras.ply')

    print(f"Point cloud saved to {output_ply}")


# Example usage:
# Assuming `depth_images` is a list/array of depth images (numpy arrays).


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <image_depth_directory> <pose_directory> <output_ply_file>")
        sys.exit(1)

    image_depth_dir = sys.argv[1]
    pose_dir = sys.argv[2]
    output_ply = sys.argv[3]

    main(image_depth_dir, pose_dir, output_ply)

"""


"""import os
import sys
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def load_images_depths(image_depth_dir):
    images = []
    depths = []
    num_images = 0

    for filename in sorted(os.listdir(image_depth_dir)):
        if filename.endswith('.png'):
            img_path = os.path.join(image_depth_dir, filename)
            depth_path = os.path.join(image_depth_dir,
                                      f"{os.path.splitext(filename)[0]}.npy")  # Assuming depth maps have the same name with .npy extension
            img = np.array(Image.open(img_path))  # Load PNG image
            depth = np.load(depth_path)  # Load depth map
            images.append(img)
            depths.append(depth)
            num_images += 1

    return images, depths, num_images


def load_poses(pose_dir, num_images):
    poses = []
    for i in range(1, num_images+1):  # Assuming 1.txt corresponds to image 0.png, 2.txt to 1.png, etc.
        print(i)
        pose = np.loadtxt(os.path.join(pose_dir, f'{i}.txt'))
        poses.append(pose)

    return poses


def save_ply_cloud(points, filename):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for point, color in points:
            f.write('{} {} {} {} {} {}\n'.format(point[0], point[1], point[2], color[0], color[1], color[2]))

    print(f"Point cloud PLY file saved: {filename}")



def create_camera_pyramid(matrix, pyramid_size=0.1):
    origin = matrix[:3, 3]
    R = matrix[:3, :3]

    # Define the pyramid base and tip
    base = pyramid_size * np.array([
        [1,  1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1,  1, 1]
    ])
    tip = np.array([0, 0, 0])

    # Apply the rotation and translation
    base_transformed = (R @ base.T).T + origin
    tip_transformed = R @ tip + origin

    # Create vertices
    vertices = np.vstack((base_transformed, tip_transformed))
    return vertices

def save_ply_cameras(cameras, filename):
    vertices = []
    vertex_idx = 0
    faces = []

    for cam in cameras:
        pose = cam['pose']
        pyramid = create_camera_pyramid(pose)

        # Add the pyramid vertices to the list
        vertices.extend(pyramid)

        # Create the faces (4 faces for the pyramid)
        tip_index = vertex_idx + 4
        for i in range(4):
            next_i = (i + 1) % 4
            faces.append([vertex_idx + i, vertex_idx + next_i, tip_index])
        vertex_idx += 5

    # Create PLY elements
    vertices_np = np.array(vertices)
    vertex_elements = [(vertex[0], vertex[1], vertex[2]) for vertex in vertices_np]
    face_elements = [(face,) for face in faces]

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    face_dtype = [('vertex_indices', 'i4', (3,))]

    vertex_element = PlyElement.describe(np.array(vertex_elements, dtype=vertex_dtype), 'vertex')
    face_element = PlyElement.describe(np.array(face_elements, dtype=face_dtype), 'face')

    ply_data = PlyData([vertex_element, face_element], text=True)
    ply_data.write(filename)

    print(f"Cameras PLY file saved: {filename}")

import matplotlib.pyplot as plt

def visualize_depth_image(depth_image):
    # Visualize depth map
    plt.figure()
    plt.imshow(depth_image, cmap='gray')
    plt.title('Depth Map')
    plt.colorbar()
    plt.axis('off')
    plt.show()


def main(image_depth_dir, pose_dir, output_ply):
    # Load images and depths from the same directory
    images, depths, num_images = load_images_depths(image_depth_dir)

    # Load camera poses
    poses = load_poses(pose_dir, num_images)

    # Camera intrinsics (pinhole model)
    fx = 355.5
    fy = 355.5
    cx = 255.5
    cy = 255.5

    point_cloud = {}
    cameras = []

    visualize_depth_image(depths[0])

    for i in range(min(num_images,50) ):
        print(i)
        img = images[i]
        depth = depths[i]
        pose = poses[i]

        height, width = img.shape[:2]
        step = max(height, width) // 50  # Step size for 50x50 grid
        for v in range(0, height, step):
            for u in range(0, width, step):
                Z = depth[v, u]
                if Z < 1 or Z > 6:  # No depth available
                    continue

                X = (u - cx) * Z / fx
                Y = (cy - v) * Z / fy

                # 3D point in camera coordinates
                P_camera = np.array([X, Y, Z, 1])

                # Transform to world coordinates
                P_world = np.dot(pose, P_camera)[:3]

                # Round to nearest 0.01 resolution in world space
                P_rounded = tuple(np.round(P_world / 0.01) * 0.01)

                # Color from RGB image
                color = img[v, u]

                # Store in dictionary with rounded position as key to avoid duplicates
                if P_rounded not in point_cloud:
                    point_cloud[P_rounded] = color

        # Save camera pose with viewing direction
        cameras.append({'pose': pose})

    # Convert dictionary to list for saving as PLY
    point_cloud_list = [(np.array(key), value) for key, value in point_cloud.items()]

    # Save point cloud to PLY with camera poses
    save_ply_cloud(point_cloud_list, output_ply)

    save_ply_cameras(cameras, 'output_ply_cameras.ply')

    print(f"Point cloud saved to {output_ply}")


# Example usage:
# Assuming `depth_images` is a list/array of depth images (numpy arrays).


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <image_depth_directory> <pose_directory> <output_ply_file>")
        sys.exit(1)

    image_depth_dir = sys.argv[1]
    pose_dir = sys.argv[2]
    output_ply = sys.argv[3]

    main(image_depth_dir, pose_dir, output_ply)

with open(os.path.join(output_dir, 'images.txt'), 'r') as file:
      lines = file.readlines()

# Step 2: Modify every second line starting from the second line
  for img_id in range(1,len(lines)//2+1):
      print(img_id, len(lines))
      img_points = []
      for p, obs in point_cloud.items():
          id = obs['id']
          for i, u ,v, _ in obs['observations']:
              if i == img_id:
                  img_points.append(f"{u} {v} {id}")
              else:
                  print('false')
      lines[2*img_id -1] = ' '.join(img_points) + '\n'

  with open(os.path.join(output_dir, 'images.txt'), 'w') as file:
      file.writelines(lines)"""
