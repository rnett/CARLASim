import gc
import os
import warnings
from pathlib import Path
from typing import Union

import ffmpeg
import h5py
import imageio
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

tf.get_logger().setLevel('ERROR')


# Returns x as a one-dimensional vector with each element
# repeated n_repeats times.
@tf.function
def _repeat(x, n_repeats: int):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


# Returns a "wrapped" version of the given coordinate.
#
# If the coord is out of bounds, it will be wrapped backwards or
# forwards to the correct location.
@tf.function
def _wrap_coords(dim, width, dim_max, dim_min=np.float32(0)):
    # wrap forward (< min)
    dim_safe = tf.where(tf.less(dim, np.float32(0)),
                        dim + width,
                        dim)
    # wrap backwards (> max)
    dim_safe = tf.where(tf.greater(dim, dim_max),
                        dim - width,
                        dim)
    return dim_safe


@tf.function
def bilinear_sampler(imgs, coords, do_wrap=True, mask=None):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
      mask: image mask indicating points to ignore
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """
    coords = tf.tile(coords, [imgs.shape[0], 1, 1, 1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.shape
        coord_size = coords.shape
        out_size = coords.shape.as_list()
        out_size[3] = imgs.shape.as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = np.float32(0)

        ## bilinear interp weights, with points outside the grid having weight 0

        if do_wrap:
            # wrap around in x-direction
            width = tf.cast(tf.shape(imgs)[2], 'float32')
            x0_safe = _wrap_coords(x0, width, dim_max=x_max)
            x1_safe = _wrap_coords(x1, width, dim_max=x_max)

            wt_x0 = x1 - coords_x
            wt_x1 = coords_x - x0
        else:
            x0_safe = tf.clip_by_value(x0, zero, x_max)
            x1_safe = tf.clip_by_value(x1, zero, x_max)

            wt_x0 = x1_safe - coords_x
            wt_x1 = coords_x - x0_safe

        # clip in y-direction (range: 0 - y_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')

        coord_size = [int(x) for x in coord_size]
        out_size = [int(x) for x in out_size]

        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## ignore pixels under the mask (weight = 0)
        if mask is not None:
            imgs = tf.math.sign(mask - 1) + imgs * mask

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')),
                          out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')),
                          out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')),
                          out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')),
                          out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])

        return output


@tf.function
def stitch_image_tensors(lut, images, depth_multiplier, mask, rgb: bool):
    lutx = tf.cast(lut[:, :, 0], 'float32')
    luty = tf.cast(lut[:, :, 1], 'float32')

    imgs = tf.cast(images, 'float32')

    lutx_re = tf.expand_dims(lutx, axis=0)
    lutx_re = tf.expand_dims(lutx_re, axis=-1)
    luty_re = tf.expand_dims(luty, axis=0)
    luty_re = tf.expand_dims(luty_re, axis=-1)

    coords = tf.concat([lutx_re, luty_re], axis=-1)

    if not rgb:
        pano = bilinear_sampler(imgs, coords, mask=mask)
    else:
        pano = bilinear_sampler(imgs, coords)

    if not rgb:
        pano = tf.squeeze(pano, -1)
        if depth_multiplier is not None:
            pano *= depth_multiplier
        return tf.clip_by_value(pano, -1, 2 ** 16 - 1)
    else:
        return pano


def read_video(file) -> np.ndarray:
    probe = ffmpeg.probe(str(file))
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    # vcodec='libx264', preset='veryslow', crf='0'

    data, _ = ffmpeg \
        .input(str(file)) \
        .output('pipe:', format='rawvideo', pix_fmt='rgb24') \
        .global_args('-loglevel', 'quiet') \
        .run(capture_stdout=True)

    return np.frombuffer(data, np.uint8).reshape([-1, height, width, 3])


def save_data(data: np.ndarray, dir: Union[Path, str], name: str, rgb: bool, samples: bool = True):
    if isinstance(dir, str):
        dir = Path(dir)
    dir = dir.absolute().resolve()

    # doesn't work for depth (its 16bit)
    if rgb:

        if samples:
            imageio.imwrite(dir / f"{name}_sample.png", data[10])

        n, height, width, channels = data.shape
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo',
                       pix_fmt='rgb24',
                       s='{}x{}'.format(width, height))
                .output(str(dir / f"{name}.mkv"),) # pix_fmt='yuv420p', # , vcodec='libx264'
                .overwrite_output()
                .global_args('-loglevel', 'quiet', "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0")
        )

        print("\nProc:", " ".join(process.compile()))

        process = process.run_async(pipe_stdin=True)

        for frame in data:
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()
    else:
        with h5py.File(str(dir / f"{name}.hdf5"), 'w') as f:
            f.create_dataset("data", data.shape, data.dtype, data,
                             compression='gzip', compression_opts=9)

        if samples:
            data = data.astype('float32')
            gc.collect()
            data = np.divide(data, np.float32(255), out=data)
            data = np.add(data, np.float32(1), out=data)
            data = np.log(data, out=data)
            data = np.subtract(data, np.min(data), out=data)
            max_scale = np.max(data)
            data = np.multiply(data, np.float32(255) / max_scale, out=data)
            data = data.astype('uint8')
            gc.collect()

            imageio.imwrite(dir / f"{name}_sample.png", data[10])
            n, height, width, channels = data.shape
            process = (
                ffmpeg
                    .input('pipe:', format='rawvideo',
                           pix_fmt='gray',
                           s='{}x{}'.format(width, height))
                    .output(str(dir / f"{name}.mkv"), pix_fmt='yuv420p', vcodec='libx264',
                            preset='veryslow', crf='0')
                    .overwrite_output()
                    .global_args('-loglevel', 'quiet')
                    .run_async(pipe_stdin=True)

            )

            for frame in data:
                process.stdin.write(frame.tobytes())

            process.stdin.close()
            process.wait()
