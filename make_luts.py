#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from carla_constants import *


def make_spherical_lut(lut_file, output_width, output_height):
    lutx = np.zeros((output_height, output_width))
    luty = np.zeros((output_height, output_width))
    depth_multiplier = np.zeros((output_height, output_width))
    local_thetas = np.zeros((output_height, output_width))
    local_phis = np.zeros((output_height, output_width))

    thetas = np.linspace(-np.pi, np.pi, num=output_width, endpoint=False)
    phis = np.linspace(-np.pi / 2, np.pi / 2, num=output_height,
                       endpoint=False)

    # CARLA parameters
    input_width = IMAGE_WIDTH
    input_height = IMAGE_HEIGHT
    fov = FOV
    focal = input_width / (2 * np.tan(fov * np.pi / 360))
    c_x = input_width * 0.5
    c_y = input_height * 0.5

    for r in tqdm(range(output_height), desc='Height', unit='pixel'):
        phi = phis[r]
        for c in range(output_width):
            theta = thetas[c]

            # get XYZ point
            Z = np.sin(theta) * np.cos(phi)
            Y = np.sin(phi)
            X = np.cos(theta) * np.cos(phi)

            # translate to x, y, mag coords for proper face
            # input/index of form: Back, Left, Front, Right, Top, Bottom
            largest = np.argmax(np.abs([X, Y, Z]))

            if largest == 0:
                if X <= 0:  # back
                    ind = 0
                    x = -Z
                    y = Y
                    mag = X
                else:  # front
                    ind = 2
                    x = Z
                    y = Y
                    mag = X
            elif largest == 1:
                if Y <= 0:  # top
                    ind = 4
                    x = Z
                    y = X
                    mag = Y
                else:  # bottom
                    ind = 5
                    x = Z
                    y = -X
                    mag = Y
            else:
                if Z <= 0:  # left
                    ind = 1
                    x = X
                    y = Y
                    mag = Z
                else:  # right
                    ind = 3
                    x = -X
                    y = Y
                    mag = Z

            # local_theta = np.arctan2(x, mag)
            # local_phi = np.arcsin(
            #     y / np.sqrt(x ** 2 + y ** 2 + mag ** 2))

            # project back to raw
            x = focal * x / np.abs(mag) + c_x
            y = focal * y / np.abs(mag) + c_y

            local_theta = np.arctan2(x - c_x, focal)
            local_phi = np.arctan2(y - c_y, focal)

            depth_multiplier[r, c] = 1 / (
                    np.cos(local_theta) * np.cos(local_phi))

            # offset for given image
            x = x + ind * input_width

            # store in lookup table
            lutx[r, c] = x
            luty[r, c] = y
            local_thetas[r, c] = local_theta
            local_phis[r, c] = local_phi

    lut = np.concatenate(
        [np.expand_dims(lutx, axis=-1), np.expand_dims(luty, axis=-1),
         np.expand_dims(depth_multiplier, axis=-1)], axis=-1)
    np.save(lut_file, lut)

    # solve for theta intrinsics
    x0 = 0
    x1 = len(thetas) - 1
    theta0 = thetas[0]
    theta1 = thetas[-1]
    c_theta = (theta0 * x1 - theta1 * x0) / (theta0 - theta1)
    f_theta = (x0 - x1) / (theta0 - theta1)

    # solve for phi intrinsics
    y0 = 0
    y1 = len(phis) - 1
    phi0 = phis[0]
    phi1 = phis[-1]
    c_phi = (phi0 * y1 - phi1 * y0) / (phi0 - phi1)
    f_phi = (y0 - y1) / (phi0 - phi1)

    intrinsics_file = Path(lut_file.name).with_name("spherical_intrinsics.txt")

    # save intrinsics to file
    with intrinsics_file.open('w') as f:
        f.write('%.15f %.15f %.15f %.15f' % (f_theta, c_theta, f_phi, c_phi))


def load_spherical_intrinsics(intrinsics_file: Path = Path("./spherical_intrinsics.txt")):
    """
    :return: (f_theta, c_theta, f_phi, c_phi)
    """
    txt = intrinsics_file.open("r").read()
    return (np.float32(t) for t in txt)


def make_cylindrical_lut(lut_file, output_width, output_height):
    bottom = -0.5
    top = 0.5

    lutx = np.zeros((output_height, output_width))
    luty = np.zeros((output_height, output_width))

    thetas = np.linspace(-np.pi, np.pi, num=output_width, endpoint=False)
    heights = np.linspace(bottom, top, num=output_height,
                          endpoint=True)

    # CARLA parameters
    input_width = IMAGE_WIDTH
    input_height = IMAGE_HEIGHT
    fov = FOV
    focal = input_width / (2 * np.tan(fov * np.pi / 360))
    c_x = input_width * 0.5
    c_y = input_height * 0.5

    for r in tqdm(range(output_height), desc='Height', unit='pixel'):
        height = heights[r]
        for c in range(output_width):
            theta = thetas[c]

            # select raw image
            if theta >= -3 * np.pi / 4 and theta < -np.pi / 4:
                ind = 1
                theta_offset = np.pi / 2  # left
            elif theta >= -np.pi / 4 and theta < np.pi / 4:
                ind = 2
                theta_offset = 0  # forward
            elif theta >= np.pi / 4 and theta < 3 * np.pi / 4:
                ind = 3
                theta_offset = -np.pi / 2  # right
            else:
                ind = 0
                theta_offset = -np.pi  # backwards

            # get XYZ point
            X = np.sin(theta + theta_offset)
            Y = height
            Z = np.cos(theta + theta_offset)

            # project to raw image
            x = focal * X / Z + c_x
            y = focal * Y / Z + c_y

            # offset for given image
            x = x + ind * input_width

            # store in lookup table
            lutx[r, c] = x
            luty[r, c] = y

    lut = np.concatenate(
        [np.expand_dims(lutx, axis=-1), np.expand_dims(luty, axis=-1)], axis=-1)
    np.save(lut_file, lut)

    # solve for theta intrinsics
    x0 = 0
    x1 = len(thetas) - 1
    theta0 = thetas[0]
    theta1 = thetas[-1]
    c_theta = (theta0 * x1 - theta1 * x0) / (theta0 - theta1)
    f_theta = (x0 - x1) / (theta0 - theta1)

    # solve for Z intrinsics
    y0 = 0
    y1 = len(heights) - 1
    Z0 = heights[0]
    Z1 = heights[-1]
    c_Z = (Z0 * y1 - Z1 * y0) / (Z0 - Z1)
    f_Z = (y0 - y1) / (Z0 - Z1)

    intrinsics_file = Path(lut_file.name).with_name("cylindrical_intrinsics.txt")

    # save intrinsics to file
    with intrinsics_file.open('w') as f:
        f.write('%.15f %.15f %.15f %.15f' % (f_theta, c_theta, f_Z, c_Z))


def load_cylindrical_intrinsics(intrinsics_file: Path = Path("./cylindrical_intrinsics.txt")):
    """
    :return: (f_theta, c_theta, f_Z, c_Z)
    """
    txt = intrinsics_file.open("r").read()
    return (np.float32(t) for t in txt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--cylindrical_lut", "-c",
                        help="Cylindrical lookup table file.")

    parser.add_argument("--spherical_lut", "-s",
                        help="Spherical lookup table file.")

    parser.add_argument("--output_width", "-ow",
                        type=int,
                        default=2048,
                        help="Output width (in pixels)")

    parser.add_argument("--output_height", "-oh",
                        type=int,
                        default=1024,
                        help="Output height (in pixels)")

    args = parser.parse_args()

    if 'cylindrical_lut' in args:
        print("Making cylindrical lookup table...")
        make_cylindrical_lut(open(args.cylindrical_lut, 'wb+'), args.output_width, args.output_height)

    if 'spherical_lut' in args:
        print("Making spherical lookup table...")
        make_spherical_lut(open(args.spherical_lut, 'wb+'), args.output_width, args.output_height)
