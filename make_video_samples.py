import gc
import numpy as np
from pathlib import Path

import ffmpeg
from tqdm import tqdm

from recordings import Recording, SplitFrame, SingleFrame
from sides import Side

recordings = [Recording.from_dir("E:/carla/town03/cloudy/noon/cars_40_peds_200_index_0"), ]

pbar = tqdm(recordings, desc="Recordings")

for recording in pbar:
    pbar.set_postfix_str(str(recording.base_data_dir))
    recording: Recording
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='gray',
                   s='{}x{}'.format(2048, 1024))
            .output(str(recording.base_data_dir / "sample_cylindrical_depth.mkv"), pix_fmt='yuv420p',
                    vcodec='libx264')
            .overwrite_output()
            .global_args('-loglevel', 'quiet', "-preset", "ultrafast", "-crf", "12")
    ).run_async(pipe_stdin=True)

    data = recording.data.cylindrical.depth[100:300].astype('float32')

    data = np.divide(data, np.float32(255), out=data)
    data = np.add(data, np.float32(1), out=data)
    data = np.log(data, out=data)
    data = np.subtract(data, np.min(data), out=data)
    max_scale = np.max(data)
    data = np.multiply(data, np.float32(255) / (max_scale / 2), out=data)
    data = data.astype('uint8')

    for frame in tqdm(data, desc="Frames", unit='frame'):
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='gray',
                   s='{}x{}'.format(2048, 1024))
            .output(str(recording.base_data_dir / "sample_spherical_depth.mkv"), pix_fmt='yuv420p',
                    vcodec='libx264')
            .overwrite_output()
            .global_args('-loglevel', 'quiet', "-preset", "ultrafast", "-crf", "12")
    ).run_async(pipe_stdin=True)

    data = recording.data.spherical.depth[100:300].astype('float32')

    data = np.divide(data, np.float32(255), out=data)
    data = np.add(data, np.float32(1), out=data)
    data = np.log(data, out=data)
    data = np.subtract(data, np.min(data), out=data)
    max_scale = np.max(data)
    data = np.multiply(data, np.float32(255) / (max_scale / 2), out=data)
    data = data.astype('uint8')

    for frame in tqdm(data, desc="Frames", unit='frame'):
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

    gc.collect()

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='{}x{}'.format(2048, 1024))
            .output(str(recording.base_data_dir / "sample_cylindrical_rgb.mkv"), pix_fmt='yuv420p',
                    vcodec='libx264')
            .overwrite_output()
            .global_args('-loglevel', 'quiet', "-preset", "ultrafast", "-crf", "12")
    ).run_async(pipe_stdin=True)

    for frame in tqdm(recording.data.cylindrical.rgb[100:300], desc="Frames", unit='frame'):
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='{}x{}'.format(2048, 1024))
            .output(str(recording.base_data_dir / "sample_spherical_rgb.mkv"), pix_fmt='yuv420p',
                    vcodec='libx264')
            .overwrite_output()
            .global_args('-loglevel', 'quiet', "-preset", "ultrafast", "-crf", "12")
    ).run_async(pipe_stdin=True)

    for frame in tqdm(recording.data.spherical.rgb[100:300], desc="Frames", unit='frame'):
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

'''
Bikes:
E:\carla\town02\clear\sunset\cars_20_peds_200_index_0
E:\carla\town02\wetcloudy\noon\cars_20_peds_200_index_1
'''

"""
ffmpeg \
  -i sample_spherical_rgb.mkv \
  -i sample_spherical_depth.mkv \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  sample_spherical.mp4
  
ffmpeg \
  -i sample_cylindrical_rgb.mkv \
  -i sample_cylindrical_depth.mkv \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  sample_cylindrical.mp4
"""
