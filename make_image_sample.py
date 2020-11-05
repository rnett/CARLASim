from imageio import imwrite

from recordings import Recording
import numpy as np

recording = Recording.from_dir("E:/carla/town02/clear/noon/cars_10_peds_200_index_3")
index = 70

def gray2rgb(im, cmap='gray'):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt  # doesn't work in docker
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, cmap='plasma'):
    depth = 1 / depth.astype('float32')
    depth = depth / np.nanmax(depth)
    # depth = gray2rgb(depth, cmap=cmap)
    return depth

def make_sample(file, color, depth):
    depth = np.squeeze(depth).astype('float32')
    depth = normalize_depth_for_display(depth)

    depth_plasma = gray2rgb(depth, "plasma") * 256

    imwrite(file, np.concatenate([color, depth_plasma], axis=0))

make_sample("spherical_sample.png", recording.data.spherical.rgb[index], recording.data.spherical.depth[index])
make_sample("cylindrical_sample.png", recording.data.cylindrical.rgb[index], recording.data.cylindrical.depth[index])
