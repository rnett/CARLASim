
# Overview

This is the simulation code used for my thesis, *Dataset and Evaluation of Self-Supervised Learning for Panoramic Depth Estimation*.

Useful scripts are `simulate.py`, `make_luts.py`, `stitch.py`, and `upload.py`, all of which have argparse help except `upload.py`.

`upload.py` usage:

Takes either a list of recording directories, or "a/all <base_dir>" ("a" or "all") to upload all.
If --remove is present, removed data files after upload.
If --reupload is present, uploads files even if they have already been uploaded (upload file is present).

Carla constants are in `carla_constants.py`.

`export.py` exports depthmaps as mesh (`.ply`) files.  The recording is hard coded in the script.
`make_image_sample.py`, `make_samples.py`, and `make_video_samples.py` are utility scripts that aren't intended for anyone else's use, but the source may be informative.
`simulate_all.py` is similar: it is used to run batches of simulations using hardcoded parameters.
