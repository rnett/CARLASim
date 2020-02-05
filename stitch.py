#!/usr/bin/env python
from abc import ABC, abstractmethod
import argparse
import gc
import os
import warnings
from pathlib import Path
from typing import Union, Generator, Tuple

import ffmpeg
import h5py
import imageio
import numpy as np

import utils
from carla_constants import *
from utils import save_data, stitch_image_tensors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tqdm import tqdm

from recordings import Recording, SplitFrame
from sides import Side


def de_batch_gen(gen: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    for a in gen:
        if np.ndim(a) == 4:
            for ar in a:
                yield ar
        elif np.ndim(a) == 3:
            yield a
        else:
            raise ValueError


def batch_gen(batch_size: int, gen: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    batch = []
    for a in de_batch_gen(gen):
        batch.append(a)

        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []

    if len(batch) > 0:
        yield np.stack(batch)


class StitchSource(ABC):

    def __init__(self, recording: Recording, rgb: bool):
        self.rgb = rgb
        self.recording = recording
        self.num_frames = len(self.recording.raw.frames)

    @property
    @abstractmethod
    def should_video(self) -> bool:
        pass

    @property
    @abstractmethod
    def video_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dtype(self) -> str:
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int, int]:
        pass

    @property
    @abstractmethod
    def get_frames(self) -> Generator[np.ndarray, None, None]:
        pass

    @property
    def dataset_name(self) -> str:
        return "rgb" if self.rgb else "depth"

    @property
    @abstractmethod
    def pbar_desc(self) -> str:
        pass


class PanoramaStitchSource(StitchSource):

    def __init__(self, recording: Recording, lut, batch_size: int, spherical: bool, rgb: bool):
        super().__init__(recording, rgb)
        self.spherical = spherical
        self.batch_size = batch_size

        if not isinstance(lut, Path):
            lut = Path(lut)

        self.lut = np.load(lut.absolute().resolve())

    @property
    def should_video(self) -> bool:
        return self.rgb

    @property
    def video_name(self) -> str:
        return "spherical.mkv" if self.spherical else "cylindrical.mkv"

    @property
    def get_frames(self) -> Generator[np.ndarray, None, None]:
        return batch_gen(100, _stich(self))

    @property
    def dtype(self) -> str:
        if self.rgb:
            return 'uint8'
        else:
            return 'uint16'

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self.num_frames, self.lut.shape[0], self.lut.shape[1], 3 if self.rgb else 1

    @property
    def pbar_desc(self) -> str:
        return f"Stitching {'spherical' if self.spherical else 'cylindrical'} {'RGB' if self.rgb else 'Depth'} frames"


class SideStitchSource(StitchSource):

    def __init__(self, recording: Recording, side: Side, rgb: bool):
        super().__init__(recording, rgb)
        self.side = side

    @property
    def should_video(self) -> bool:
        return self.side is Side.Front and self.rgb

    @property
    def video_name(self) -> str:
        return "front.mkv"

    @property
    def get_frames(self) -> Generator[np.ndarray, None, None]:
        yield np.stack(list(self._frames()))

    def _frames(self):
        data = r.raw[side]

        for frame in data.frames:
            if self.rgb:
                yield frame.rgb_data
            else:
                yield frame.depth_data[:, :, np.newaxis]

    @property
    def dtype(self) -> str:
        if self.rgb:
            return 'uint8'
        else:
            return 'uint16'

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self.num_frames, IMAGE_WIDTH, IMAGE_HEIGHT, 3 if self.rgb else 1

    @property
    def pbar_desc(self) -> str:
        return f"Collecting {self.side.name.lower()} frames"


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


def _stich(source: PanoramaStitchSource):
    im_type = source.dtype
    if source.rgb:
        im_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
    else:
        im_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 1)

    if source.spherical:
        concat_shape = (im_shape[0], im_shape[1] * 6, im_shape[2])
    else:
        concat_shape = (im_shape[0], im_shape[1] * 4, im_shape[2])

    frames = source.recording.raw.frames

    if not source.spherical:
        # TODO enabling this messes with height? / adds depth stripes even w/ height.  Maybe set in LUT creation like spherical
        size = source.lut.shape[1] // 4
        t = np.pi / 4
        thetas = np.linspace(-t, t, size)
        all_thetas = np.concatenate((thetas[size // 2:],
                                     thetas,
                                     thetas,
                                     thetas,
                                     thetas[:size // 2]))
        depth_multiplier = 1 / np.cos(all_thetas)
        depth_multiplier = depth_multiplier[np.newaxis, np.newaxis, :]
        depth_multiplier = depth_multiplier.astype('float32')
        # depth_multiplier = np.float32(1)
    else:
        depth_multiplier = source.lut[:, :, 2].astype('float32')

    for batch in _batch_frames(frames, source.batch_size, concat_shape, im_type,
                               source.spherical, source.rgb):

        if not source.rgb:
            mask = np.logical_not(batch == 10000).astype(
                'float32')  # 10000 or 9999?  # np.isclose(batch, 10000, atol=0, rtol=0)
        else:
            mask = None

        batch_frames = np.array(
            stitch_image_tensors(source.lut[:, :, 0:2],
                                 batch,
                                 depth_multiplier,
                                 mask,
                                 source.rgb)).astype(im_type)

        # if mask is not None:
        #     batch_frames[batch_frames == 0] = -1

        if not source.rgb:
            batch_frames = batch_frames[:, :, :, np.newaxis]

        yield batch_frames


IMAGE_SAVE_POINT = 200


def _process(source: StitchSource, create_in):
    shape = source.shape
    dtype = source.dtype

    dataset = create_in.create_dataset(source.dataset_name, shape=shape, dtype=dtype,
                                       compression='gzip', compression_opts=4,
                                       # compression='lzf',
                                       shuffle=True,
                                       fletcher32=True)

    if source.should_video:
        video_file = r.base_data_dir / source.video_name

        if video_file.exists():
            video_file.unlink()

        if video_file.with_suffix(".png").exists():
            video_file.with_suffix(".png").unlink()

        n, height, width, channels = shape
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo',
                       pix_fmt='rgb24',
                       s='{}x{}'.format(width, height))
                .output(str(video_file), pix_fmt='yuv420p',
                        vcodec='libx264')
                .overwrite_output()
                .global_args('-loglevel', 'quiet', "-preset", "ultrafast", "-crf", "12")
        )

        process = process.run_async(pipe_stdin=True)

    pbar = tqdm(
        desc=source.pbar_desc,
        unit='frame',
        total=shape[0],
        mininterval=0)

    i = 0
    for frame in source.get_frames:
        if np.ndim(frame) == 4:
            # multiple frames

            dataset[i:i + len(frame), :, :, :] = frame

            if source.should_video:
                for f in frame:
                    process.stdin.write(f.tobytes())

                if i <= IMAGE_SAVE_POINT < i + len(frame):
                    imageio.imwrite(str(video_file.with_suffix(".png")), frame[IMAGE_SAVE_POINT - i])

            i += len(frame)
            pbar.update(len(frame))

        elif np.ndim(frame) == 3:
            # single frame

            dataset[i, :, :, :] = frame

            if source.should_video:
                process.stdin.write(frame.tobytes())

                if i == IMAGE_SAVE_POINT:
                    imageio.imwrite(str(video_file.with_suffix(".png")), frame)

            i += 1
            pbar.update(1)
        else:
            raise ValueError(frame)

    if source.should_video:
        process.stdin.close()
        process.wait()

    gc.collect()


"""
Stats:
    Gzip level 5: 2092s, 8.17 GB
    No compression: 1345.46s, 36 GB
    Gzip level 2: 1524.21s, 8.9 GB
    Gzip level 3: 1724.21s, 8.4 GB
    Gzip level 4: 1796.92s, 8.4 GB      ** best so far
    Gzip level 9: 9658.69s, 7.67 GB
    LZF: 1261.98s, 13.5 GB
"""

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

    parser.add_argument("--no_cylindrical",
                        action='store_true',
                        help="Skip cylindrical stitching (and lut if "
                             "specified).")

    parser.add_argument("--no_spherical",
                        action='store_true',
                        help="Skip spherical stitching (and lut if specified).")

    parser.add_argument("--overwrite", "-o",
                        action='store_true',
                        help="Skip spherical stitching (and lut if specified).")

    parser.add_argument("--do_uploaded", "-u",
                        action='store_true',
                        help="Re-stitch even if it has been uploaded (./uploaded exists)")

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

            cylindrical_file = r.base_data_dir / "cylindrical.hdf5"
            spherical_file = r.base_data_dir / "spherical.hdf5"
            pinhole_file = r.base_data_dir / "pinhole.hdf5"

            if r.is_uploaded and not args.do_uploaded:
                print(f"Stitched data for {r.base_data_dir} has already been uploaded, skipping")
                continue

            if cylindrical_file.exists() or spherical_file.exists() or pinhole_file.exists():
                if args.overwrite:
                    if cylindrical_file.exists():
                        cylindrical_file.unlink()
                    if spherical_file.exists():
                        spherical_file.unlink()
                    if pinhole_file.exists():
                        pinhole_file.unlink()
                elif args.all is not None:
                    if cylindrical_file.exists() and spherical_file.exists() and pinhole_file.exists():
                        print(f"Data already exists, skipping: {r.base_data_dir}")
                        continue
                    else:
                        raise FileExistsError(f"Some data exists for {r.base_data_dir}, but not all of it")
                else:
                    raise FileExistsError("Data already exists")

            if not args.no_cylindrical:
                with h5py.File(str(cylindrical_file), "w") as file:
                    _process(PanoramaStitchSource(r, args.cylindrical_lut, args.batch_size, False, True), file)
                    _process(PanoramaStitchSource(r, args.cylindrical_lut, args.batch_size, False, False), file)

            if not args.no_spherical:
                with h5py.File(str(spherical_file), "w") as file:
                    _process(PanoramaStitchSource(r, args.spherical_lut, args.batch_size, True, True), file)
                    _process(PanoramaStitchSource(r, args.spherical_lut, args.batch_size, True, False), file)

            # save pinhole frames in matching formats

            with h5py.File(str(pinhole_file), "w") as file:
                for side in tqdm(list(Side), desc="Saving sides", unit="side", total=len(list(Side))):
                    side_group = file.create_group(side.name.lower())

                    _process(SideStitchSource(r, side, True), side_group)
                    _process(SideStitchSource(r, side, False), side_group)
