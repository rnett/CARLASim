import trimesh as trimesh

from recordings import Recording, SingleRecordingDataset
import numpy as np

recording = Recording.from_dir("E:/carla/town03/cloudy/noon/cars_40_peds_200_index_0")

def get_spherical_export(data: SingleRecordingDataset, outfile, frame: int = 200):
    fx, cx, fy, cy = np.loadtxt("./spherical_intrinsics.txt", delimiter=' ')

    rgb = data.rgb[frame][:]
    depth = data.depth[frame][:, :, 0]
    depth[depth == 0] = 1

    img_height = rgb.shape[0]
    img_width = rgb.shape[1]

    py, px = np.meshgrid(range(img_height), range(img_width), indexing='ij')
    py = (py - cy) / fy
    px = (px - cx) / fx
    spherical_coords = np.stack([px, py], axis=0)
    X = np.sin(spherical_coords[0]) * np.cos(spherical_coords[1])
    Y = np.sin(spherical_coords[1])
    Z = np.cos(spherical_coords[0]) * np.cos(spherical_coords[1])
    cam_coords = np.stack([X, Y, Z], axis=2) * depth[:, :, None]

    # get vertices and colors
    x = cam_coords[:, :, 0]
    y = cam_coords[:, :, 1]
    z = cam_coords[:, :, 2]
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    vertex_colors = np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1)

    # make triangle faces
    def get_index(r, c):
        return r * img_width + (c % img_width)

    faces = []
    for r in range(img_height - 1):
        for c in range(img_width):
            faces.append([get_index(r, c), get_index(r + 1, c), get_index(r, c + 1)])
            faces.append([get_index(r + 1, c), get_index(r + 1, c + 1), get_index(r, c + 1)])
    faces = np.stack(faces, axis=0)

    # export using trimesh
    T = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors, faces=faces)
    T.export(outfile)


def get_cylindrical_export(data: SingleRecordingDataset, outfile, frame: int = 200):
    fx, cx, fy, cy = np.loadtxt("./cylindrical_intrinsics.txt", delimiter=' ')


    rgb = data.rgb[frame][:]
    depth = data.depth[frame][:, :, 0]
    depth[depth == 0] = 1

    img_height = rgb.shape[0]
    img_width = rgb.shape[1]

    py, px = np.meshgrid(range(img_height), range(img_width), indexing='ij')
    py = (py - cy) / fy
    px = (px - cx) / fx
    cylinder_coords = np.stack([px, py], axis=0)
    X = np.sin(cylinder_coords[0])
    Y = cylinder_coords[1]
    Z = np.cos(cylinder_coords[0])
    cam_coords = np.stack([X, Y, Z], axis=2) * depth[:, :, None]

    # get vertices and colors
    x = cam_coords[:, :, 0]
    y = cam_coords[:, :, 1]
    z = cam_coords[:, :, 2]
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    vertex_colors = np.stack([red.flatten(), green.flatten(), blue.flatten()], axis=1)

    # make triangle faces
    def get_index(r, c):
        return r * img_width + (c % img_width)

    faces = []
    for r in range(img_height - 1):
        for c in range(img_width):
            faces.append([get_index(r, c), get_index(r + 1, c), get_index(r, c + 1)])
            faces.append([get_index(r + 1, c), get_index(r + 1, c + 1), get_index(r, c + 1)])
    faces = np.stack(faces, axis=0)

    # export using trimesh
    T = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors, faces=faces)
    T.export(outfile)


get_spherical_export(recording.data.spherical, "spherical_mesh.ply")
get_cylindrical_export(recording.data.cylindrical, "cylindrical_mesh.ply")
