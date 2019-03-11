import pathlib
import pandas as pd
import numpy as np


class KEYS:
    PHANTOM_OUT_FILE = "phantomD.bin"
    DEFAULT_MATERIAL_FILE = "range_material_phantomD.dat"
    DEFAULT_ACTIVITY_FILE = "activity_range_phantomD.dat"
    MACSET = pathlib.Path("/mnt/gluster/qinglong/macset")


def range_material(gray_scale, work_dir):
    np.trim_zeros(gray_scale)
    gray_scale = gray_scale.tolist()
    gray_scale.append(0)
    gray_scale.reverse()
    gray_scale = np.array(gray_scale, dtype=np.float32)
    rm_header = pd.DataFrame([str(len(gray_scale))])
    rm_body = pd.DataFrame([gray_scale, gray_scale, ['Air' for _ in range(len(gray_scale))]]).T

    with open(work_dir/KEYS.DEFAULT_MATERIAL_FILE, "w") as f:
        rm_header.to_csv(f, header=False, index=False, sep='\t', mode='a')
        rm_body.to_csv(f, header=False, index=False, sep='\t', mode='a')
        print(f"Default material file wirten in {str(work_dir/KEYS.DEFAULT_MATERIAL_FILE)}")
    return work_dir/KEYS.DEFAULT_MATERIAL_FILE


def activity_range(gray_scale, work_dir):
    gray_scale = gray_scale.tolist()
    gray_scale = np.trim_zeros(gray_scale)
    gray_scale.append(0)
    gray_scale.reverse()

    x1 = gray_scale.copy()
    x1.insert(0, str(len(gray_scale)))

    x2 = gray_scale.copy()
    x2.insert(0, np.nan)

    x3 = gray_scale.copy()
    x3 = [int(i * 10) for i in x3]
    x3.insert(0, np.nan)

    range_a = pd.DataFrame([x1, x2, x3]).T

    with open(work_dir/KEYS.DEFAULT_ACTIVITY_FILE, 'w') as f:
        range_a.to_csv(f, header=False, index=False, sep='\t', mode='a')
        print(f"Default activity file wirten in {str(work_dir/KEYS.DEFAULT_ACTIVITY_FILE)}")
    return work_dir/KEYS.DEFAULT_ACTIVITY_FILE


def phantom_D(phantom, num_layers, work_dir):
    output_phantom_size = list(phantom.shape)
    output_phantom_size.insert(0, 10)
    canvas = np.zeros(output_phantom_size, dtype=np.float32)
    for i in range(num_layers):
        canvas[i] = phantom
    canvas.tofile(str(work_dir/"phantomD.bin"))
    return work_dir/"phantomD.bin"
