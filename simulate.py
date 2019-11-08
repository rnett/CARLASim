#!/usr/bin/env python
import argparse
import glob
import os
import signal
import subprocess
import sys
import time
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


def launch_server(carla_script: str, extra_args: List[str]):
    args = [carla_script]
    args.extend(extra_args)
    args.append("-quality-level=Epic")
    return subprocess.Popen(args, preexec_fn=os.setsid)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run CARLA simulation and save pinhole images")

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

    parser.add_argument("sunset",
                        default=False,
                        type=bool,
                        nargs='?',
                        help="Whether to simulate at sunset.")

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

    parser.add_argument("--ticks", "-t",
                        default=20 * 1000,
                        type=int,
                        help="Number of ticks to run the simulation for (an "
                             "image is saved every 20 ticks).")

    parser.add_argument("--seed",
                        default=123,
                        type=int,
                        help="RNG Seed")

    parser.add_argument("--no_overwrite",
                        dest='overwrite',
                        action='store_false',
                        help="If present, don't overwrite an existing output")

    args = parser.parse_args()

    if args.carla != "":
        server = launch_server(args.carla, args.carla_args)
        time.sleep(5)
    else:
        server = None

    try:

        city = None
        for c in list(City):
            if args.city.lower() == c.name.lower():
                city = c
                break

        if city is None:
            print(
                f"Given city {args.city} is not a valid city.  Try one of "
                f"{[c.name for c in list(City)]}")

        rain = None
        for r in list(Rain):
            if args.rain.lower() == r.name.lower():
                rain = r
                break

        if rain is None:
            print(
                f"Given rain status {args.city} is not a valid rain status.  "
                f"Try one of {[r.name for r in list(Rain)]}")

        car_idx = None
        if 'car_idx' in args:
            car_idx = args.car_idx

        sim = CarlaSim(
            SimConfig(args.cars, args.pedestrians, city, rain,
                      args.sunset),
            base_output_folder=args.output_dir,
            seed=args.seed,
            host=str(args.host),
            port=str(args.port),
            car_idx=car_idx,
            overwrite=args.overwrite)

    except Exception as e:
        if server is not None:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        raise e

    try:
        for i in tqdm(range(args.ticks), desc="Ticks", unit="ticks"):
            sim.tick()
        sim.end()
    finally:
        sim.end()
        if server is not None:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        quit()
