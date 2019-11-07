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


def make_spherical_lut(lut_file, intrinsics_file=None):
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
    focal = input_width / (2 * tan(fov * np.pi / 360))
    c_x = input_width * 0.5
    c_y = input_height * 0.5

    for r in range(output_height):
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

            # project back to pinhole
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


def _stich_spherical(recording: Recording, lut: Union[str, Path], rgb: bool):
    if not isinstance(lut, Path):
        lut = Path(lut)

    global_min = carla_sim.DEPTH_MULTIPLIER

    lut = lut.absolute().resolve()

    lut = np.load(lut)
    lutx = lut[:, :, 0].astype('float32')
    luty = lut[:, :, 1].astype('float32')
    depth_multiplier = lut[:, :, 2].astype('float32')

    if rgb:
        im_shape = (carla_sim.IMAGE_WIDTH, carla_sim.IMAGE_HEIGHT, 3)
        im_type = 'uint8'
    else:
        im_shape = (carla_sim.IMAGE_WIDTH, carla_sim.IMAGE_HEIGHT, 1)
        im_type = 'uint16'

    concat_shape = (im_shape[0], im_shape[1] * 6, im_shape[2])
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
    pano = bilinear_sampler(imgs, coords,
                            mask=imgs == carla_sim.DEPTH_MULTIPLIER)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        for frame in tqdm(recording.pinhole.frames,
                          desc=f"Stitching spherical "
                               f"{'RGB' if rgb else 'Depth'}",
                          unit='frame',
                          ):
            frame: SplitFrame
            if rgb:
                imB = frame[Side.Back].rgb_data
                imF = frame[Side.Front].rgb_data
                imL = frame[Side.Left].rgb_data
                imR = frame[Side.Right].rgb_data
                imT = frame[Side.Top].rgb_data
                imBo = frame[Side.Bottom].rgb_data
                im = np.concatenate([imB, imL, imF, imR, imT, imBo], axis=1)
            else:
                imB = frame[Side.Back].depth_data
                imF = frame[Side.Front].depth_data
                imL = frame[Side.Left].depth_data
                imR = frame[Side.Right].depth_data
                imT = frame[Side.Top].depth_data
                imBo = frame[Side.Bottom].depth_data
                im = np.concatenate([imB, imL, imF, imR, imT, imBo], axis=1)
                im = np.expand_dims(im, -1)

            outfile = recording.spherical.file_for(frame.frame, not rgb)

            res = sess.run(pano, {im_ph: im, lutx_ph: lutx, luty_ph: luty})
            res = np.squeeze(res, axis=0)
            if not rgb:
                res = res.squeeze()
                res = res * depth_multiplier
                res = res.clip(0, 2 ** 16 - 1)  # must fit in uint16

            if res.min() < global_min:
                global_min = res.min()

            res = res.astype(im_type)

            # save
            imageio.imwrite(outfile, res)

    # print("Global min (should be > 1 for scaling):", global_min)


import tensorflow as tf


@tf.function
def _stitch_image_tensors(lut: tf.Tensor, images: tf.Tensor):
    lutx = tf.cast(lut[:, :, 0], 'float32')
    luty = tf.cast(lut[:, :, 1], 'float32')
    depth_multiplier = tf.cast(lut[:, :, 2], 'float32')

    imgs = tf.cast(images, 'float32')

    lutx_re = tf.expand_dims(lutx, axis=0)
    lutx_re = tf.expand_dims(lutx_re, axis=-1)
    luty_re = tf.expand_dims(luty, axis=0)
    luty_re = tf.expand_dims(luty_re, axis=-1)

    coords = tf.concat([lutx_re, luty_re], axis=-1)
    pano = bilinear_sampler(imgs, coords,
                            mask=imgs == carla_sim.DEPTH_MULTIPLIER)
    pano *= depth_multiplier
    return tf.clip_by_value(pano, 0, 2 ** 16 - 1)


def stitch_spherical(recording: Recording, lut: Union[str, Path], parent):
    _stich_spherical(recording, lut, True, parent)
    _stich_spherical(recording, lut, False, parent)
