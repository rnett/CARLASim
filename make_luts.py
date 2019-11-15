#!/usr/bin/env python
import argparse

import numpy as np
from tqdm import tqdm

import carla_sim


def make_spherical_lut(lut_file):
    output_width = 1520
    output_height = 760

    lutx = np.zeros((output_height, output_width))
    luty = np.zeros((output_height, output_width))
    depth_multiplier = np.zeros((output_height, output_width))
    local_thetas = np.zeros((output_height, output_width))
    local_phis = np.zeros((output_height, output_width))

    thetas = np.linspace(-np.pi, np.pi, num=output_width, endpoint=False)
    phis = np.linspace(-np.pi / 2, np.pi / 2, num=output_height,
                       endpoint=False)

    # CARLA parameters
    input_width = carla_sim.IMAGE_WIDTH
    input_height = carla_sim.IMAGE_HEIGHT
    fov = carla_sim.FOV
    focal = input_width / (2 * np.tan(fov * np.pi / 360))
    c_x = input_width * 0.5
    c_y = input_height * 0.5

    for r in tqdm(range(output_height), desc='Height', unit='pixel'):
        phi = phis[r]
        for c in tqdm(range(output_width), desc='Width', unit='pixel'):
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


def make_cylindrical_lut(lut_file):
    output_width = 1520
    output_height = 760

    bottom = -0.5
    top = 0.5

    lutx = np.zeros((output_height, output_width))
    luty = np.zeros((output_height, output_width))

    thetas = np.linspace(-np.pi, np.pi, num=output_width, endpoint=False)
    heights = np.linspace(bottom, top, num=output_height,
                          endpoint=True)

    # CARLA parameters
    input_width = carla_sim.IMAGE_WIDTH
    input_height = carla_sim.IMAGE_HEIGHT
    fov = carla_sim.FOV
    focal = input_width / (2 * np.tan(fov * np.pi / 360))
    c_x = input_width * 0.5
    c_y = input_height * 0.5

    for r in tqdm(range(output_height), desc='Height', unit='pixel'):
        height = heights[r]
        for c in tqdm(range(output_width), desc='Width', unit='pixel'):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--cylindrical_lut", "-c",
                        help="Cylindrical lookup table file.")

    parser.add_argument("--spherical_lut", "-s",
                        help="Spherical lookup table file.")

    args = parser.parse_args()

    if 'cylindrical_lut' in args:
        print("Making cylindrical lookup table...")
        make_cylindrical_lut(open(args.cylindrical_lut, 'wb+'))

    if 'spherical_lut' in args:
        print("Making spherical lookup table...")
        make_spherical_lut(open(args.spherical_lut, 'wb+'))
