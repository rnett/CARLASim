import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

tf.get_logger().setLevel('ERROR')


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

    # Returns x as a one-dimensional vector with each element
    # repeated n_repeats times.
    def _repeat(x, n_repeats):
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
    def _wrap_coords(dim, width, dim_max, dim_min=np.float32(0)):
        # wrap forward (< min)
        dim_safe = tf.where(tf.less(dim, zero),
                            dim + width,
                            dim)
        # wrap backwards (> max)
        dim_safe = tf.where(tf.greater(dim, dim_max),
                            dim - width,
                            dim)
        return dim_safe

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

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
            imgs = imgs * mask

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
