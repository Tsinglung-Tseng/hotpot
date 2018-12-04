import os
import pathlib
import shutil
from pathlib import Path
from typing import Iterable, Dict, List


def phantom_sub_dir(i, type="derenzo"):
    WORKDIR = pathlib.Path('/mnt/gluster/qinglong/DLSimu')
    return WORKDIR/f"{type}_phantom_{i}"


def scale_sub_dir(i, j, type="derenzo"):
    return phantom_sub_dir(i, type)/f"mac_sub{j}"


def recon_depends(sub_id):
    return pathlib.Path(f'/mnt/gluster/qinglong/recon/x{sub_id}')


def simu_depends(sub_id):
    return pathlib.Path(f'/mnt/gluster/qinglong/macset/mac_sub{sub_id}')


def get_not_done(dir_to_check: Iterable[Path]) -> Dict:
    root = []
    sinogram = []

    for d in dir_to_check:
        if not Path.is_file(d/"result.root"):
            root.append(d)

        if not Path.is_file(d/"sinogram.s"):
            sinogram.append(d)

    return {"root": root,
            "sinogram": sinogram}


def get_done(dir_to_check: Iterable[Path]) -> Dict:
    root = []
    sinogram = []

    for d in dir_to_check:
        if Path.is_file(d/"result.root"):
            root.append(d)

        if Path.is_file(d/"sinogram.s"):
            sinogram.append(d)

    return {"root": root,
            "sinogram": sinogram}


def get_all_workdirs(phantom_range: Iterable[int], sub_range: Iterable[int], type) -> List:
    all_dir = []

    for i in phantom_range:
        for j in sub_range:
            if Path.is_dir(scale_sub_dir(i, j, type)):
                all_dir.append(scale_sub_dir(i, j, type))

    return all_dir


def load_recon_depends(work_dir: Path):
    sub_id = int(work_dir.name[-1])
    print(f"loading recon dependencies from x{sub_id}, at {recon_depends(sub_id)}")
    for p in (recon_depends(sub_id)).iterdir():
        shutil.copyfile(p, work_dir/p.name)


def load_simu_depends(work_dir: Path):
    sub_id = int(work_dir.name[-1])
    print(f"loading simulation dependencies from mac_sub{sub_id}, at {simu_depends(sub_id)}")
    for p in (simu_depends(sub_id)).iterdir():
        shutil.copyfile(p, work_dir/p.name)


def make_dir(work_dir: Path):
    try:
        os.makedirs(work_dir)
        print(f"Directory {work_dir} added.")
    except FileExistsError:
        print(f"Directory {work_dir} exists.")