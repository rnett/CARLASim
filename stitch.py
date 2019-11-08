#!/usr/bin/env python
import argparse
import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from utils import save_video, stitch_image_tensors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tqdm import tqdm

import carla_sim
from recordings import Recording
from sides import Side


def _batch_frames(frames, batch_size, im_shape, im_dtype, spherical: bool,
                  rgb: bool):
    batch_i = 0
    batch = np.empty(shape=(batch_size,) + im_shape, dtype=im_dtype)
    for i in range(len(frames)):
        frame = frames[i]
        if rgb:
            imB = frame[Side.Back].rgb_data
            imF = frame[Side.Front].rgb_data
            imL = frame[Side.Left].rgb_data
            imR = frame[Side.Right].rgb_data
            if spherical:
                imT = frame[Side.Top].rgb_data
                imBo = frame[Side.Bottom].rgb_data
                im = np.concatenate([imB, imL, imF, imR, imT, imBo], axis=1)
            else:
                im = np.concatenate([imB, imL, imF, imR], axis=1)
        else:
            imB = frame[Side.Back].depth_data
            imF = frame[Side.Front].depth_data
            imL = frame[Side.Left].depth_data
            imR = frame[Side.Right].depth_data
            if spherical:
                imT = frame[Side.Top].depth_data
                imBo = frame[Side.Bottom].depth_data
                im = np.concatenate([imB, imL, imF, imR, imT, imBo], axis=1)
            else:
                im = np.concatenate([imB, imL, imF, imR], axis=1)
            im = np.expand_dims(im, -1)
        batch[batch_i] = im

        if batch_i == batch_size - 1:
            yield batch
            batch = np.empty(shape=(batch_size,) + im_shape, dtype=im_dtype)
            batch_i = 0
        else:
            batch_i += 1

    if batch_i > 0:
        yield batch[:batch_i]


def _stich(recording: Recording, lut: Union[str, Path], batch_size,
           spherical: bool, rgb: bool):
    if not isinstance(lut, Path):
        lut = Path(lut)

    lut = lut.absolute().resolve()

    lut = np.load(lut)

    if rgb:
        im_shape = (carla_sim.IMAGE_WIDTH, carla_sim.IMAGE_HEIGHT, 3)
        im_type = 'uint8'
    else:
        im_shape = (carla_sim.IMAGE_WIDTH, carla_sim.IMAGE_HEIGHT, 1)
        im_type = 'uint16'

    if spherical:
        concat_shape = (im_shape[0], im_shape[1] * 6, im_shape[2])
    else:
        concat_shape = (im_shape[0], im_shape[1] * 4, im_shape[2])

    frames = recording.pinhole.frames

    pbar = tqdm(
        desc=f"Stitching {'spherical' if spherical else 'cylindrical'} "
             f"{'RGB' if rgb else 'Depth'}",
        unit='frame',
        total=len(frames),
        mininterval=0)

    frame_num = 0
    if rgb:
        video_frames = np.empty(shape=(len(frames), 760, 1520, 3),
                                dtype=im_type)
    else:
        video_frames = np.empty(shape=(len(frames), 760, 1520), dtype=im_type)

    if not spherical:
        size = im_shape[1] // 4
        t = np.pi / 4
        thetas = np.linspace(-t, t, size)
        all_thetas = np.concatenate((thetas[size // 2:],
                                     thetas,
                                     thetas,
                                     thetas,
                                     thetas[:size // 2]))
        depth_multiplier = 1 / np.cos(all_thetas)
        depth_multiplier = depth_multiplier[np.newaxis, :, np.newaxis]
        depth_multiplier = depth_multiplier.astype('float32')
    else:
        depth_multiplier = lut[:, :, 2].astype('float32')

    for batch in _batch_frames(frames, batch_size, concat_shape, im_type,
                               spherical, rgb):
        batch_frames = np.array(
            stitch_image_tensors(lut[:, :, 0:2],
                                 batch,
                                 depth_multiplier,
                                 rgb)).astype(
            im_type)
        pbar.update(len(batch_frames))
        video_frames[frame_num:frame_num + len(batch_frames)] = batch_frames
        frame_num += len(batch_frames)

    if not rgb:
        video_frames = video_frames[:, :, :, np.newaxis]

    print("Saving...\r")
    save_video(video_frames,
               (
                   recording.spherical.data_dir if spherical else
                   recording.cylindrical.data_dir)
               / f"{'rgb' if rgb else 'depth'}.mkv",
               rgb)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Stitch pinhole images into cylindrical and spherical "
                    "panoramas.")

    parser.add_argument("cylindrical_lut",
                        help="Cylindrical lookup table file.")

    parser.add_argument("spherical_lut",
                        help="Spherical lookup table file.")

    parser.add_argument("--single", "-s",
                        action='append',
                        help="Stitch files in this single directory.  Can be "
                             "specified multiple times.")

    parser.add_argument("--all", "-a",
                        help="Stitch all files for all recordings in this "
                             "base directory.")

    parser.add_argument("--batch_size", "-b",
                        type=int,
                        default=30,
                        help="Batch size used for stitching")

    parser.add_argument("--remake_luts",
                        action='store_true',
                        help="Remake lut files.")

    parser.add_argument("--no_cylindrical",
                        action='store_true',
                        help="Skip cylindrical stitching (and lut if "
                             "specified).")

    parser.add_argument("--no_spherical",
                        action='store_true',
                        help="Skip spherical stitching (and lut if specified).")

    args = parser.parse_args()

    if 'single' in args:
        recordings = [Recording.from_dir(d) for d in args.single]
    elif 'all' in args:
        recordings = Recording.all_in_dir(args.all)
    else:
        print(
            "No recording directories specified.  Use --all/-a or --single/-s")
        quit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # TODO detect existing

        parent = tqdm(recordings, desc="Recordings", unit='recording')

        for r in parent:
            if not args.no_cylindrical:
                _stich(r, args.cylindrical_lut, args.batch_size, False, True)
                _stich(r, args.cylindrical_lut, args.batch_size, False, False)

            if not args.no_spherical:
                _stich(r, args.spherical_lut, args.batch_size, True, True)
                _stich(r, args.spherical_lut, args.batch_size, True, False)
