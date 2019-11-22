#!/usr/bin/env python
import argparse
import gc
import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np

import utils
from utils import save_video, stitch_image_tensors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tqdm import tqdm

import carla_sim
from recordings import Recording, SplitFrame
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

    frames = recording.raw.frames

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

        if not rgb:
            mask = np.logical_not(batch == 10000).astype('float32')  # np.isclose(batch, 10000, atol=0, rtol=0)
        else:
            mask = None

        batch_frames = np.array(
            stitch_image_tensors(lut[:, :, 0:2],
                                 batch,
                                 depth_multiplier,
                                 mask,
                                 rgb)).astype(
            im_type)

        # if mask is not None:
        #     batch_frames[batch_frames == 0] = -1

        pbar.update(len(batch_frames))
        video_frames[frame_num:frame_num + len(batch_frames)] = batch_frames
        frame_num += len(batch_frames)

    if not rgb:
        video_frames = video_frames[:, :, :, np.newaxis]

    save_video(video_frames,
               (
                   recording.spherical.data_dir if spherical else
                   recording.cylindrical.data_dir)
               / f"{'rgb' if rgb else 'depth'}.mkv",
               rgb)
    return video_frames


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Stitch raw images into cylindrical and spherical "
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

    if args.single is not None:
        recordings = [Recording.from_dir(d) for d in args.single]
    elif args.all is not None:
        recordings = Recording.all_in_dir(args.all)
    else:
        print(
            "No recording directories specified.  Use --all/-a or --single/-s")
        quit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # TODO detect existing

        pbar = tqdm(recordings, desc="Recordings", unit='recording')
        for r in pbar:
            r: Recording
            pbar.set_postfix_str("Recording: " + str(r.base_data_dir))

            if not args.no_cylindrical:
                _stich(r, args.cylindrical_lut, args.batch_size, False, True)
                gc.collect()
                _stich(r, args.cylindrical_lut, args.batch_size, False, False)
                gc.collect()

            if not args.no_spherical:
                _stich(r, args.spherical_lut, args.batch_size, True, True)
                gc.collect()
                _stich(r, args.spherical_lut, args.batch_size, True, False)
                gc.collect()

            # save pinhole frames in matching formats

            for side in tqdm(list(Side), desc="Saving sides", unit="side", total=len(list(Side))):

                r.pinhole_data_dir.mkdir(exist_ok=True)

                first_rgb = r.raw[side].frames[0].rgb_data
                rgb_frames = np.empty(shape=(len(r.raw.frames),) + first_rgb.shape, dtype=first_rgb.dtype)
                rgb_frames[0] = first_rgb

                for i in tqdm(range(1, len(r.raw.frames)), desc=f"Collecting {side} rgb frames"):
                    rgb_frames[i] = r.raw[side].frames[i].rgb_data

                utils.save_video(rgb_frames, r.pinhole_data_dir / f"{side.name.lower()}_rgb.mkv", True)

                del rgb_frames
                gc.collect()

                # now do depth

                first_depth = r.raw[side].frames[0].depth_data

                depth_frames = np.empty(shape=(len(r.raw.frames),) + first_depth.shape, dtype=first_depth.dtype)
                depth_frames[0] = first_depth

                for i in tqdm(range(1, len(r.raw.frames)), desc=f"Collecting {side} depth frames"):
                    depth_frames[i] = r.raw[side].frames[i].depth_data

                utils.save_video(depth_frames[:, :, :, np.newaxis],
                                 r.pinhole_data_dir / f"{side.name.lower()}_depth.mkv", False)

                del depth_frames
                gc.collect()
