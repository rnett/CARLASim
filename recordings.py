from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Union

import h5py
import imageio
import numpy as np

import utils
from config import SimConfig
from sides import Side


class SingleFrame:
    def __init__(self, frame: int, rgb: Path, depth: Path):
        self.frame = frame
        self.rgb_file = rgb
        self.depth_file = depth

    @property
    def rgb_data(self) -> np.ndarray:
        return imageio.imread(self.rgb_file)[:, :, :-1]

    @property
    def depth_data(self) -> np.ndarray:
        return imageio.imread(self.depth_file)


class SplitFrame:
    def __init__(self, frame: int, rgb: Dict[Side, Path],
                 depth: Dict[Side, Path]):
        self.frame = frame
        self.rgb_files = rgb
        self.depth_files = depth

    @property
    def rgb_data(self) -> Dict[Side, np.ndarray]:
        return {k: imageio.imread(v)[:, :, :-1] for k, v in self.rgb_files.items()}

    @property
    def depth_data(self) -> Dict[Side, np.ndarray]:
        return {k: imageio.imread(v) for k, v in self.depth_files.items()}

    def __getitem__(self, side: Side) -> SingleFrame:
        return SingleFrame(self.frame, self.rgb_files[side],
                           self.depth_files[side])


class RecordingData:
    def __init__(self, data_dir: Path):
        if not data_dir.exists():
            data_dir.mkdir()

        self.data_dir = data_dir
        self.files = [f for f in data_dir.iterdir() if f.is_file()]
        super().__init__()


class RawRecordingData(RecordingData):
    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self._frames = None

    @property
    def frames(self) -> List[SplitFrame]:
        if self._frames is None:
            rgb = {}
            depth = {}

            for f in self.files:

                frame = int(f.name.split('_')[1])
                side = None
                side_name = f.name.split('_')[0].capitalize()
                for s in list(Side):
                    if s.name == side_name:
                        side = s
                        break

                if side is None:
                    raise ValueError(f"Unknown side {f.name.split('_')[0]}")

                if f.name.endswith('rgb.png'):
                    if frame not in rgb:
                        rgb[frame] = {}

                    rgb[frame][side] = f
                else:
                    if frame not in depth:
                        depth[frame] = {}

                    depth[frame][side] = f

            incomplete_rgb = [k for k, v in rgb.items() if
                              any(s not in v for s in list(Side))]
            incomplete_depth = [k for k, v in depth.items() if
                                any(s not in v for s in list(Side))]

            if len(incomplete_rgb) > 0 or len(incomplete_depth) > 0:
                raise ValueError(
                    f"Incomplete frames (missing sides): incomplete rgb "
                    f"frames: {incomplete_rgb}, incomplete depth frames: "
                    f"{incomplete_depth}")

            if rgb.keys() != depth.keys():
                missing_depth = [i for i in rgb.keys() if i not in depth]
                missing_rgb = [i for i in depth.keys() if i not in rgb]
                raise ValueError(
                    f"Mismatched frame indices: missing depth for frames "
                    f"{missing_depth}, missing rbg for frames {missing_rgb}")

            rgb = list(rgb.items())
            depth = list(depth.items())

            rgb.sort(key=lambda x: x[0])
            depth.sort(key=lambda x: x[0])

            rgb = [x[1] for x in rgb]
            depth = [x[1] for x in depth]

            self._frames = [SplitFrame(i, rgb, depth) for i, (rgb, depth) in
                            enumerate(zip(rgb, depth))]

        return self._frames

    def __iter__(self) -> Iterator[SplitFrame]:
        return iter(self.frames)

    def __getitem__(self, item: Side):
        return SingleSideRawRecordingData(item, self)

    def file_for(self, side: Side, frame: int, depth: bool = False) -> Path:
        return self.data_dir / f"{side.name.lower()}_{frame}_" \
                               f"{'depth' if depth else 'rgb'}.png"


class SingleSideRawRecordingData(RecordingData):

    def __init__(self, side: Side, all_sides: RawRecordingData):
        super().__init__(all_sides.data_dir)
        self._frames = [x[side] for x in all_sides.frames]

    @property
    def frames(self) -> List[SingleFrame]:
        return self._frames

    def __iter__(self) -> Iterator[SingleFrame]:
        return iter(self.frames)


class StitchedRecordingData(RecordingData):
    def __init__(self, data_dir: Path, prefix: str = ""):
        super().__init__(data_dir)
        self.prefix = prefix

    @property
    def rgb_data(self) -> np.ndarray:
        return utils.read_video(self.data_dir / f"{self.prefix}rgb.mkv")

    @property
    def depth_data(self) -> np.ndarray:
        with h5py.File(str(self.data_dir / f"{self.prefix}depth.hdf5"), 'r') as f:
            data = f["data"][:]
        return data


class PinholeStitchedRecordingData(RecordingData):
    def __init__(self, data_dir: Path):
        super().__init__(data_dir)

    def __getitem__(self, side: Side) -> StitchedRecordingData:
        return StitchedRecordingData(self.data_dir, side.name.lower() + "_")

    @property
    def back(self) -> StitchedRecordingData:
        return self[Side.Back]

    @property
    def front(self) -> StitchedRecordingData:
        return self[Side.Front]

    @property
    def left(self) -> StitchedRecordingData:
        return self[Side.Left]

    @property
    def right(self) -> StitchedRecordingData:
        return self[Side.Right]

    @property
    def top(self) -> StitchedRecordingData:
        return self[Side.Top]

    @property
    def bottom(self) -> StitchedRecordingData:
        return self[Side.Bottom]


class Recording:
    def __init__(self, base_dir: Union[str, Path], config: SimConfig):
        self.config = config

        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)

        self.base_dir = base_dir.absolute().resolve()

        self.base_data_dir = self.base_dir / config.folder_name
        self.raw_data_dir = self.base_data_dir / "raw"
        self.pinhole_data_dir = self.base_data_dir / "pinhole"
        self.spherical_data_dir = self.base_data_dir / "spherical"
        self.cylindrical_data_dir = self.base_data_dir / "cylindrical"

        self._raw = None
        self._pinhole = None
        self._spherical = None
        self._cylindrical = None

    def __repr__(self):
        return repr(self.base_data_dir)

    def __str__(self):
        return str(self.base_data_dir)

    @property
    def raw(self) -> RawRecordingData:
        if self._raw is None:
            self._raw = RawRecordingData(self.raw_data_dir)

        return self._raw

    @property
    def pinhole(self) -> PinholeStitchedRecordingData:
        if self._pinhole is None:
            self._pinhole = PinholeStitchedRecordingData(self.pinhole_data_dir)

        return self._pinhole

    @property
    def spherical(self) -> StitchedRecordingData:
        if self._spherical is None:
            self._spherical = StitchedRecordingData(self.spherical_data_dir)

        return self._spherical

    @property
    def cylindrical(self) -> StitchedRecordingData:
        if self._cylindrical is None:
            self._cylindrical = StitchedRecordingData(self.cylindrical_data_dir)

        return self._cylindrical

    @staticmethod
    def all_in_dir(base_dir: Union[Path, str], fail: bool = False) -> List:
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)

        base_dir = base_dir.absolute().resolve()

        # dirs is all town folders
        dirs = list(base_dir.iterdir())
        # dirs is all rain folders
        dirs = [f for d in dirs for f in d.iterdir()]
        # dirs is all time folders
        dirs = [f for d in dirs for f in d.iterdir()]
        # dirs is all run folders
        dirs = [f for d in dirs for f in d.iterdir()]

        configs = []

        for d in dirs:
            name = str(d.absolute().resolve().relative_to(base_dir).as_posix())
            try:
                configs.append(SimConfig.from_folder_name(name))
            except Exception as e:
                print(
                    f"Could not convert {d} to config. Used string \""
                    f"{name}\"\n Exception {e}")

                if fail:
                    raise e

        return [Recording(base_dir, c) for c in configs]

    @staticmethod
    def from_dir(dir: Union[str, Path]):
        if not isinstance(dir, Path):
            dir = Path(dir)

        dir = dir.absolute().resolve()

        base_dir = dir.parent.parent.parent.parent
        name = str(dir.absolute().resolve().relative_to(base_dir).as_posix())

        try:
            config = SimConfig.from_folder_name(name)
        except Exception as e:
            print(
                f"Could not convert {dir} to config. Used string \""
                f"{name}\"\n Exception {e}")
            raise e

        return Recording(base_dir, config)
