import glob
import os
import pathlib
import subprocess
import shutil
from hotpot.simulation.sinogramSR import x1TOx4, x2TOx4
import pathlib
import numpy as np


class KEYS:

    @classmethod
    def PHANTOM_SUB_DIR(cls, i):
        return f"derenzo_phantom_{i}"

    @classmethod
    def ReconDepends(cls, sub=1):
        return pathlib.Path(f'/mnt/gluster/qinglong/recon/x{sub}')

    WORKDIR = pathlib.Path('/mnt/gluster/qinglong/DLSimu')


def load_recon_depends(sub_workdir):
    sub_id = int(sub_workdir.name[-1])
    if sub_id in [1, 2, 3, 4]:
        print(f"    loading recon dependencies from x{sub_id}, at {KEYS.ReconDepends(sub_id)}")
        for p in (KEYS.ReconDepends(sub_id)).iterdir():
            shutil.copyfile(p, sub_workdir / p.name)


def run_recon():
    print("    Run root2bin")
    subprocess.run(["bash", "root2bintxt.sh"], cwd=str(sub_workdir))

    print("    Run txt2h5")
    subprocess.run(["python", "txt2h5.py"], cwd=str(sub_workdir))

    print("    Run run_stir")
    subprocess.run(["bash", "run_stir.sh"], cwd=str(sub_workdir))


def x1_to_x4(WORKDIR):
    S = np.fromfile(str(WORKDIR / 'sinogram.s'), dtype=np.float32)
    SR = S.reshape(100, 80, 80)

    SR_x1TOx4 = x1TOx4(SR)
    (np.array(list(SR_x1TOx4.values()), dtype=np.float32)).tofile("sinogram_x1_to_x4.s")


def x2_to_x4(WORKDIR):
    S = np.fromfile(str(WORKDIR / 'sinogram.s'), dtype=np.float32)
    SR_x2 = S.reshape(400, 160, 160)

    SR_x2TOx4 = x2TOx4(SR_x2)
    (np.array(list(SR_x2TOx4.values()), dtype=np.float32)).tofile("sinogram_x2_to_x4.s")


for INDEX in range(50, 60):
    task_workdir = KEYS.WORKDIR / KEYS.PHANTOM_SUB_DIR(INDEX)

    for i in range(1, 4):
        sub_workdir = task_workdir / f'mac_sub{i}'
        if os.path.isfile(sub_workdir / "result.root"):
            print(f"Yo, root in {task_workdir.name}/{sub_workdir.name}")

            load_recon_depends(sub_workdir)
            run_recon()
            if i == 1:
                print("    x1_to_x4")
                x1_to_x4(sub_workdir)
            if i == 2:
                print("    x2_to_x4")
                x2_to_x4(sub_workdir)

        else:
            print(f"Ooops, no root in {task_workdir.name}/{sub_workdir.name}")