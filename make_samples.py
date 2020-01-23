import gc
from pathlib import Path

import ffmpeg
from tqdm import tqdm

from recordings import Recording, SplitFrame, SingleFrame
from sides import Side

recordings = Recording.all_in_dir(Path("E:/carla"))

pbar = tqdm(recordings, desc="Recordings")

for recording in pbar:
    pbar.set_postfix_str(str(recording.base_data_dir))
    recording: Recording
    process_front = (
        ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='{}x{}'.format(768, 768))
            .output(str(recording.base_data_dir / "sample_front.mkv"), pix_fmt='yuv420p',
                    vcodec='libx264')
            .overwrite_output()
            .global_args('-loglevel', 'quiet', "-preset", "ultrafast", "-crf", "12")
    )
    process_back = (
        ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='{}x{}'.format(768, 768))
            .output(str(recording.base_data_dir / "sample_back.mkv"), pix_fmt='yuv420p',
                    vcodec='libx264')
            .overwrite_output()
            .global_args('-loglevel', 'quiet', "-preset", "ultrafast", "-crf", "12")
    )

    process_front = process_front.run_async(pipe_stdin=True)
    process_back = process_back.run_async(pipe_stdin=True)

    for frame in tqdm(recording.raw.frames[::5], desc="Frames", unit='frame'):
        frame: SplitFrame
        process_front.stdin.write(frame[Side.Front].rgb_data.tobytes())
        process_back.stdin.write(frame[Side.Back].rgb_data.tobytes())

    process_front.stdin.close()
    process_front.wait()

    process_back.stdin.close()
    process_back.wait()

    gc.collect()

'''
Bikes:
E:\carla\town02\clear\sunset\cars_20_peds_200_index_0
E:\carla\town02\wetcloudy\noon\cars_20_peds_200_index_1
'''