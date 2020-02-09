#!/usr/bin/env python
import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from carla_sim import CarlaSim
from config import Rain, SimConfig, City


class FramesMismatchError(Exception):
    pass


def launch_server(carla_script: str, extra_args: List[str], port=None):
    args = [carla_script]
    args.extend(extra_args)
    args.append("-quality-level=Epic")
    args.append("-opengl")
    if port is not None:
        args.append("-carla-port=" + str(port))
    return subprocess.Popen(args, preexec_fn=os.setsid)


class SimulateArgs:
    def __init__(self, city: City, cars: int, pedestrians: int, rain: Rain = Rain.Clear, sunset: bool = False,
                 car_idx: int = None, output_dir: str = '/data/carla/',
                 carla: str = "/home/rnett/carla/CARLA_0.9.6/CarlaUE4.sh", carla_args: List[str] = [""],
                 host: str = 'localhost', port='2000', frames: int = 1000, seed: int = 123,
                 overwrite: bool = True, index: int = 0):

        if not isinstance(port, str):
            port = str(port)

        self.index = index
        self.overwrite = overwrite
        self.seed = seed
        self.frames = frames
        self.port = port
        self.host = host
        self.carla_args = carla_args
        self.carla = carla
        self.car_idx = car_idx
        self.output_dir = output_dir
        self.sunset = sunset
        self.rain = rain
        self.pedestrians = pedestrians
        self.cars = cars
        self.city = city


def simulate(args: SimulateArgs):
    config = SimConfig(args.cars, args.pedestrians, args.city, args.rain,
                       args.sunset, args.index)

    output_folder = Path(args.output_dir) / config.folder_name

    if output_folder.exists():
        if args.overwrite:
            shutil.rmtree(output_folder)
        else:
            if (output_folder / "raw").exists():
                num_frames = len(list((output_folder / "raw/frames").iterdir()))

                if num_frames != 2 * 6 * args.frames:
                    raise FramesMismatchError(
                        f"Output exists in {output_folder} but has the wrong number of frames (has {num_frames}, "
                        f"needs {6 * args.frames})")
                print(
                    f"Output folder {output_folder} already exists and "
                    f"'overwrite' is false.")
                raise FileExistsError
            else:
                shutil.rmtree(output_folder)

    if args.carla != "":
        server = launch_server(args.carla, args.carla_args, args.port)
        time.sleep(10)
    else:
        server = None

    try:
        sim = CarlaSim(
            config,
            base_output_folder=args.output_dir,
            seed=args.seed,
            host=str(args.host),
            port=str(args.port),
            car_idx=args.car_idx,
            overwrite=args.overwrite)

    except Exception as e:
        if server is not None:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        raise e

    try:
        pbar = tqdm(total=args.frames, desc="Sim to " + str(output_folder), unit="frames")
        for i in range(args.frames * 20):
            if sim.tick():
                pbar.update()
        # sim.end()
    finally:
        # sim.end()
        if server is not None:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run CARLA simulation and save raw images")

    parser.add_argument("city",
                        choices=[c.name for c in list(City)],
                        help="City map to simulate on.")

    parser.add_argument("cars",
                        type=int,
                        help="The number of other cars in the simulation.")

    parser.add_argument("pedestrians",
                        type=int,
                        help="The number of pedestrians in the simulation.")

    parser.add_argument("rain",
                        default='Clear',
                        nargs='?',
                        choices=[r.name for r in list(Rain)],
                        help="Rain weather to simulate with.")

    parser.add_argument("time",
                        default='Noon',
                        type=str,
                        choices=["Noon", "Sunset"],
                        nargs='?',
                        help="Time of the simulation: can be Noon or Sunset.")

    parser.add_argument("--car_idx",
                        type=int,
                        help="Index of car blueprint to use as main car.  "
                             "Random if unspecified (based on seed).")

    parser.add_argument("--output_dir",
                        default="/data/carla/",
                        help="Base directory for carla output (Actual output "
                             "directory will be in this one, but depend on "
                             "arguments).")

    parser.add_argument("--carla",
                        default="/home/rnett/carla/CARLA_0.9.6/CarlaUE4.sh",
                        help="Carla run script/executable.  CarlaUE4.sh on "
                             "linux.  If empty, server is not started and is "
                             "assumed to already be running at given host and "
                             "port.")

    parser.add_argument("--carla-args",
                        default="",
                        nargs='*',
                        help="Additional arguments for the carla run script.")

    parser.add_argument("--host",
                        default='localhost',
                        help="Host to look for carla server at.")

    parser.add_argument("--port",
                        default=2000,
                        type=int,
                        help='Port to look for carla server at.')

    parser.add_argument("--frames", "-f",
                        type=int,
                        nargs='?',
                        help="Number of frames to run the simulation for ('frame' == image saved, a frame is saved "
                             "every 20 ticks).  If --ticks/-t is also present, must be match (frames * 20 == ticks).")

    parser.add_argument("--ticks", "-t",
                        type=int,
                        nargs='?',
                        help="Number of ticks to run the simulation for (an "
                             "image is saved every 20 ticks).")

    parser.add_argument("--seed",
                        default=123,
                        type=int,
                        help="RNG Seed")

    parser.add_argument("--index",
                        default=0,
                        type=int,
                        help="Index, used to differentiate multiple runs with the same settings.")

    parser.add_argument("--no_overwrite",
                        dest='overwrite',
                        action='store_false',
                        help="If present, don't overwrite an existing output")

    args = parser.parse_args()

    city = None
    for c in list(City):
        if args.city.lower() == c.name.lower():
            city = c
            break

    if city is None:
        print(
            f"Given city {city} is not a valid city.  Try one of "
            f"{[c.name for c in list(City)]}")

    rain = None
    for r in list(Rain):
        if args.rain.lower() == r.name.lower():
            rain = r
            break

    if rain is None:
        print(
            f"Given rain status {city} is not a valid rain status.  "
            f"Try one of {[r.name for r in list(Rain)]}")

    car_idx = None
    if 'car_idx' in args:
        car_idx = car_idx

    if args.frames is not None and args.ticks is not None:
        frames = args.frames

        if frames != args.ticks / 20:
            print(
                f"Frames and ticks both specified, must match.  Got {frames} frames ({frames * 20} ticks) "
                f"and {args.ticks} ticks")
            quit(1)
    elif args.frames is not None:
        frames = args.frames
    elif args.ticks is not None:

        if args.ticks % 20 != 0:
            print("Ticks must be divisible by 20.")

        frames = int(args.ticks / 20)
    else:
        frames = 1000

    simulate(SimulateArgs(city, args.cars, args.pedestrians, rain, args.time == "Sunset", car_idx, args.output_dir, args.carla,
             args.carla_args, args.host, args.port, frames, args.seed, args.overwrite, args.index))
