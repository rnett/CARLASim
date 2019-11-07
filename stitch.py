#!/usr/bin/env python
import argparse
import os
import warnings

from tqdm import tqdm

from recordings import Recording
from stitch_cylindrical import make_cylindrical_lut, stitch_cylindrical
from stitch_spherical import make_spherical_lut, stitch_spherical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

tf.get_logger().setLevel('ERROR')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Stitch pinhole images into cylindrical and spherical "
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

    if 'single' in args:
        recordings = [Recording.from_dir(d) for d in args.single]
    elif 'all' in args:
        recordings = Recording.all_in_dir(args.all)
    else:
        print(
            "No recording directories specified.  Use --all/-a or --single/-s")
        quit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if not os.path.exists(args.cylindrical_lut) or (
                args.remake_luts and not args.no_cylindrical):
            print("Making cylindrical lookup table...")
            make_cylindrical_lut(open(args.cylindrical_lut, 'wb+'))

        if not os.path.exists(args.spherical_lut) or (
                args.remake_luts and not args.no_spherical):
            print("Making spherical lookup table...")
            make_spherical_lut(open(args.spherical_lut, 'wb+'))

        # TODO detect existing

        parent = tqdm(recordings, desc="Recordings", unit='recording')

        for r in parent:
            if not args.no_cylindrical:
                stitch_cylindrical(r, args.cylindrical_lut, parent)

            if not args.no_spherical:
                stitch_spherical(r, args.spherical_lut, parent)
