from sys import argv

import s3fs
from tqdm import tqdm

from recordings import Recording

if __name__ == '__main__':
    args = argv[1:]

    remove = False
    if "--remove" in args:
        remove = True
        args.remove("--remove")

    re_upload = False
    if "--reupload" in args:
        re_upload = True
        args.remove("--reupload")

    if args[0] == "a" or args[0] == "all":
        recordings = Recording.all_in_dir(args[1])
    else:
        recordings = [Recording.from_dir(d) for d in args]

    fs = s3fs.S3FileSystem()

    base_dir = "s3://cscdatasets/jventu09/carla_dataset/"

    pbar = tqdm(recordings, desc="Recordings")
    for r in pbar:
        if r.is_uploaded and not re_upload:
            pbar.update()
            continue
        # if r.is_uploaded:
        #     cylindrical_file = r.base_data_dir / "cylindrical.hdf5"
        #     spherical_file = r.base_data_dir / "spherical.hdf5"
        #     pinhole_file = r.base_data_dir / "pinhole.hdf5"
        #     cylindrical_file.unlink()
        #     spherical_file.unlink()
        #     pinhole_file.unlink()


        pbar.set_postfix_str(f"Recording: {r.base_data_dir}")
        cylindrical_file = r.base_data_dir / "cylindrical.hdf5"
        spherical_file = r.base_data_dir / "spherical.hdf5"
        pinhole_file = r.base_data_dir / "pinhole.hdf5"

        if cylindrical_file.exists() and spherical_file.exists() and pinhole_file.exists():
            fs.put(str(cylindrical_file), base_dir + r.config.folder_name + "/cylindrical.hdf5")
            fs.put(str(spherical_file), base_dir + r.config.folder_name + "/spherical.hdf5")
            fs.put(str(pinhole_file), base_dir + r.config.folder_name + "/pinhole.hdf5")
            fs.put(str(r.raw_data_dir / "seed.txt"), base_dir + r.config.folder_name + "/seed.txt")
            fs.put(str(r.raw_data_dir / "pose.hdf5"), base_dir + r.config.folder_name + "/pose.hdf5")

            if remove:
                cylindrical_file.unlink()
                spherical_file.unlink()
                pinhole_file.unlink()

            (r.base_data_dir / "uploaded").touch(exist_ok=True)
        else:
            print(f"Missing some stitched data in {r.base_data_dir}, skipping")
            pbar.update()
            continue
