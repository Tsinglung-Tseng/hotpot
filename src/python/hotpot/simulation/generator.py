import pathlib
import pandas as pd
import numpy as np


class KEYS:
    PHANTOM_OUT_FILE = "phantomD.bin"
    DEFAULT_MATERIAL_FILE = "range_material_phantomD.dat"
    DEFAULT_ACTIVITY_FILE = "activity_range_phantomD.dat"
    MACSET = pathlib.Path("/mnt/gluster/qinglong/macset")


def range_material(gray_scale, work_dir):
    gray_scale.append(0).reverse()
    rm_header = pd.DataFrame(str(len(gray_scale) + 1))
    rm_body = pd.DataFrame([gray_scale, gray_scale, ['Air', 'Air', 'Air']]).T

    with open(work_dir/KEYS.DEFAULT_MATERIAL_FILE, "w") as f:
        rm_header.to_csv(f, header=False, index=False, sep='\t', mode='a')
        rm_body.to_csv(f, header=False, index=False, sep='\t', mode='a')
        print(f"Default material file wirten in {str(work_dir/KEYS.DEFAULT_MATERIAL_FILE)}")


def activity_range(gray_scale, work_dir):
    gray_scale.append(0).reverse()
    range_a = pd.DataFrame([gray_scale.insert(0, "3")],
                           [gray_scale.insert(0, np.nan)],
                           [10*i for i in gray_scale].insert(0, np.nan)).T

    with open(work_dir/KEYS.DEFAULT_ACTIVITY_FILE, 'w') as f:
        range_a.to_csv(f, header=False, index=False, sep='\t', mode='a')
        print(f"Default activity file wirten in {str(work_dir/KEYS.DEFAULT_ACTIVITY_FILE)}")


def phantom_D(phantom, work_dir):
    phantom.tofile(str(work_dir))
