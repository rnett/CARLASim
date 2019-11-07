import os
import warnings
from math import tan
from pathlib import Path
from typing import Union

import imageio
import numpy as np

from utils import bilinear_sampler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

tf.get_logger().setLevel('ERROR')

from tqdm import tqdm

import carla_sim
from recordings import Recording, SplitFrame
from sides import Side

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_cylindrical_lut(lut_file, intrinsics_file=None):
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
    focal = input_width / (2 * tan(fov * np.pi / 360))
    c_x = input_width * 0.5
    c_y = input_height * 0.5

    for r in range(output_height):
        height = heights[r]
        for c in range(output_width):
            theta = thetas[c]

            # select pinhole image
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

            # project to pinhole image
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
    # x0 = 0
    # x1 = len(thetas) - 1
    # theta0 = thetas[0]
    # theta1 = thetas[-1]
    # c_theta = (theta0 * x1 - theta1 * x0) / (theta0 - theta1)
    # f_theta = (x0 - x1) / (theta0 - theta1)
    #
    # # solve for Z intrinsics
    # y0 = 0
    # y1 = len(heights) - 1
    # Z0 = heights[0]
    # Z1 = heights[-1]
    # c_Z = (Z0 * y1 - Z1 * y0) / (Z0 - Z1)
    # f_Z = (y0 - y1) / (Z0 - Z1)
    #
    # print('Intrinsics:')
    # print('  f_theta: {}'.format(f_theta))
    # print('  c_theta: {}'.format(c_theta))
    # print('  f_Z: {}'.format(f_Z))
    # print('  c_Z: {}'.format(c_Z))
    #
    # # save intrinsics to file
    # if intrinsics_file is not None:
    #     with open(intrinsics_file, 'w') as f:
    #         f.write('%.15f %.15f %.15f %.15f' % (f_theta, c_theta, f_Z, c_Z))


def correct_planar_depth(depth):
    """
    Corrects a depth pano generated from planar depth groundtruths.

    When a cylindrical panorama is stitched from depth images given in respect
    to a plane, the resulting depth panorama is incorrect. This method scales
    the depth by 1/cos(theta), where theta is the angle from the original image
    center.

    Note: This assumes the depth is stitched from four images, each with pi/4
    fov, and is shifted pi/8 to face forward.
    """
    size = depth.shape[1] // 4
    t = np.pi / 4
    thetas = np.linspace(-t, t, size)
    all_thetas = np.concatenate((thetas[size // 2:],
                                 thetas,
                                 thetas,
                                 thetas,
                                 thetas[:size // 2]))
    fixed = depth / np.cos(all_thetas)
    return fixed


def _stich_cylindrical(recording: Recording, lut: Union[str, Path], rgb: bool,
                       parent):
    if not isinstance(lut, Path):
        lut = Path(lut)

    global_min = carla_sim.DEPTH_MULTIPLIER

    lut = lut.absolute().resolve()

    lut = np.load(lut)
    lutx = lut[:, :, 0].astype('float32')
    luty = lut[:, :, 1].astype('float32')

    if rgb:
        im_shape = (carla_sim.IMAGE_WIDTH, carla_sim.IMAGE_HEIGHT, 3)
        im_type = 'uint8'
    else:
        im_shape = (carla_sim.IMAGE_WIDTH, carla_sim.IMAGE_HEIGHT, 1)
        im_type = 'uint16'

    concat_shape = (im_shape[0], im_shape[1] * 4, im_shape[2])
    im_ph = tf.placeholder(dtype=im_type, shape=concat_shape)
    imgs = tf.cast(im_ph, 'float32')
    imgs = tf.expand_dims(imgs, axis=0)

    lutx_ph = tf.placeholder('float32', lutx.shape)
    luty_ph = tf.placeholder('float32', luty.shape)

    lutx_re = tf.expand_dims(lutx_ph, axis=0)
    lutx_re = tf.expand_dims(lutx_re, axis=-1)
    luty_re = tf.expand_dims(luty_ph, axis=0)
    luty_re = tf.expand_dims(luty_re, axis=-1)

    coords = tf.concat([lutx_re, luty_re], axis=-1)
    pano = bilinear_sampler(imgs, coords)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        for frame in tqdm(recording.pinhole.frames,
                          desc=f"Stitching cylindrical "
                               f"{'RGB' if rgb else 'Depth'}",
                          unit='frame'):
            frame: SplitFrame
            if rgb:
                imB = frame[Side.Back].rgb_data
                imF = frame[Side.Front].rgb_data
                imL = frame[Side.Left].rgb_data
                imR = frame[Side.Right].rgb_data
                im = np.concatenate([imB, imL, imF, imR], axis=1)
            else:
                imB = frame[Side.Back].depth_data
                imF = frame[Side.Front].depth_data
                imL = frame[Side.Left].depth_data
                imR = frame[Side.Right].depth_data
                im = np.concatenate([imB, imL, imF, imR], axis=1)
                im = np.expand_dims(im, -1)

            outfile = recording.cylindrical.file_for(frame.frame, not rgb)

            res = sess.run(pano, {im_ph: im, lutx_ph: lutx, luty_ph: luty})
            res = np.squeeze(res, axis=0)
            if not rgb:
                res = res.squeeze()
                res = correct_planar_depth(res)
                res = res.clip(0, 2 ** 16 - 1)  # must fit in uint16

            if res.min() < global_min:
                global_min = res.min()

            res = res.astype(im_type)

            # save
            imageio.imwrite(outfile, res)

    # print("Global min (should be > 1 for scaling):", global_min)


def stitch_cylindrical(recording: Recording, lut: Union[str, Path], parent):
    _stich_cylindrical(recording, lut, True, parent)
    _stich_cylindrical(recording, lut, False, parent)
