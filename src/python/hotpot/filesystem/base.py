import pathlib
from pathlib import Path
from typing import Iterable, Dict


def phantom_sub_dir(i):
    WORKDIR = pathlib.Path('/mnt/gluster/qinglong/DLSimu')
    return WORKDIR/f"derenzo_phantom_{i}"


def scale_sub_dir(i, j):
    return phantom_sub_dir(i)/f"mac_sub{j}"


def get_not_done(check_range: Iterable[int]) -> Dict:
    root_not_done = []
    sinogram_not_done = []

    for i in check_range:
        for j in range(1, 4):
            if not Path.is_file(scale_sub_dir(i, j) / "result.root"):
                root_not_done.append(scale_sub_dir(i, j))

            if not Path.is_file(scale_sub_dir(i, j) / "sinogram.s"):
                sinogram_not_done.append(str(scale_sub_dir(i, j)))

    return {"root_not_done": root_not_done,
            "sinogram_not_done": sinogram_not_done}