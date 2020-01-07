import imageio

import carla_sim
from recordings import Recording, SingleRecordingDataset
import numpy as np

recording = Recording.from_dir("E:/carla/town01/clear/noon/cars_50_peds_200_index_0")

def header(num_points):
    return f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

def get_spherical_export(data: SingleRecordingDataset, frame: int = 200):
    rgb_frame = data.rgb[frame][:]
    depth_frame = data.depth[frame][:, :, 0]

    height = rgb_frame.shape[0]
    width = rgb_frame.shape[1]

    out = ""

    thetas = np.tile(np.linspace(-np.pi, np.pi, num=width, endpoint=False), [height, 1])
    phis = np.tile(np.linspace(-np.pi / 2, np.pi / 2, num=height,
                               endpoint=False)[:, np.newaxis], [1, width])

    Xs = np.sin(thetas) * np.cos(phis) * depth_frame
    Ys = np.sin(phis) * depth_frame
    Zs = np.cos(thetas) * np.cos(phis) * depth_frame

    for y in range(height):
        for x in range(width):
            color = rgb_frame[y, x]
            red = color[0]
            green = color[1]
            blue = color[2]

            X = Xs[y, x]
            Y = Ys[y, x]
            Z = Zs[y, x]

            out += f"{X} {Y} {Z} {red} {green} {blue}\n"

    return out


def get_cylindrical_export(data: SingleRecordingDataset, frame: int = 200):
    rgb_frame = data.rgb[frame][:]
    depth_frame = data.depth[frame][:, :, 0]

    height = rgb_frame.shape[0]
    width = rgb_frame.shape[1]

    out = ""

    thetas = np.tile(np.linspace(-np.pi, np.pi, num=width, endpoint=False), [height, 1])
    heights = np.tile(np.linspace(-0.5, 0.5, num=height,
                                  endpoint=True)[:, np.newaxis], [1, width])

    Xs = np.sin(thetas) * depth_frame
    Zs = np.cos(thetas) * depth_frame

    #TODO is wrong
    Ys = heights * np.sqrt(np.square(Xs) + np.square(Zs))

    for y in range(height):
        for x in range(width):
            color = rgb_frame[y, x]
            red = color[0]
            green = color[1]
            blue = color[2]

            X = Xs[y, x]
            Y = Ys[y, x]
            Z = Zs[y, x]

            out += f"{X} {Y} {Z} {red} {green} {blue}\n"

    return out


with recording.data as data:
    spherical = get_spherical_export(data.spherical)
    cylindrical = get_cylindrical_export(data.cylindrical)

    num_points = 1024 * 2048

    open("spherical_mesh.ply", 'w').write(header(num_points) + spherical)
    open("cylindrical_mesh.ply", 'w').write(header(num_points) + cylindrical)

    open("both_mesh.ply", 'w').write(header(num_points * 2) + spherical + cylindrical)
