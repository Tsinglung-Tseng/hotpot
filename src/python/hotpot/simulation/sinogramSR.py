import numpy as np
from collections import Counter
import subprocess
import shutil
import pathlib


def ring_(NUM_RINGS=10):
    """
    Get ring_difference_per_segment & matrix_size for given number of rings.

    return
        ring_difference_per_segment: [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9]
        matrix_size: [10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    """
    listOfRing = np.linspace(1, NUM_RINGS, NUM_RINGS)
    ring_gap = (listOfRing - i for i in listOfRing)

    C = []
    for l in ring_gap:
        C.append(Counter(l))

    cs = sum(C, Counter())

    sorted_cs = sorted(cs.items(), key=lambda kv: kv[1], reverse=True)

    ring_difference_per_segment = [int(i[0]) for i in sorted_cs]

    matrix_size = [i[1] for i in sorted_cs]

    return [-i for i in ring_difference_per_segment], matrix_size


def name_ring_diff(num_rings, diff):
    """
    name_ring_diff(10,1)
    [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9)]

    name_ring_diff(10,-1)
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    """
    if abs(diff) > num_rings:
        raise ValueError
    if diff < 0:
        return list(zip(np.linspace(1, num_rings, num_rings, dtype=np.int16),
                        np.linspace(1, num_rings - abs(diff), num_rings - abs(diff), dtype=np.int16) - diff))
    if diff >= 0:
        return list(zip(np.linspace(1 + diff, num_rings, num_rings - diff, dtype=np.int16),
                        np.linspace(1, num_rings, num_rings, dtype=np.int16)))


def matrix_size2ind(matrix_size):
    """SR index segmentation."""
    r = []
    for i in range(0, len(matrix_size) + 1):
        r.append(sum(matrix_size[:i]))
    return list(zip(r, r[1:]))


def SRMarking(SR, matrix_ind, ring_difference_per_segment, num_rings=10):
    SR_marked = {}
    for i in range(len(matrix_ind)):
        tmp = SR[matrix_ind[i][0]:matrix_ind[i][1], :, :]
        nrd = name_ring_diff(num_rings, ring_difference_per_segment[i])
        for ii in range(len(nrd)):
            SR_marked[nrd[ii]] = tmp[ii]
    return SR_marked


def SR_mapping(lr_key):
    a, b = lr_key
    return (2 * a - 1, 2 * b - 1), (2 * a - 1, 2 * b), (2 * a, 2 * b), (2 * a, 2 * b - 1)


def resize(img, scale=2):
    img_x2 = np.zeros([img.shape[0] * scale, img.shape[1] * scale])

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_x2[x * scale, y * scale] = img[x, y] / 16 ** (scale - 1)
            img_x2[x * scale, y * scale + 1] = img[x, y] / 16 ** (scale - 1)
            img_x2[x * scale + 1, y * scale + 1] = img[x, y] / 16 ** (scale - 1)
            img_x2[x * scale + 1, y * scale] = img[x, y] / 16 ** (scale - 1)
    return img_x2


def singram_resize(SR_marked, SR_x2_marked):
    for lr_key in SR_marked.keys():
        for sr_key in SR_mapping(lr_key):
            SR_x2_marked[sr_key] = resize(SR_marked[lr_key])
    return SR_x2_marked


def x1TOx4(SR):
    # 给原始数据（x1）做标记
    ring_difference_per_segment, matrix_size = ring_(10)
    matrix_ind = matrix_size2ind(matrix_size)
    SR_marked = SRMarking(SR, matrix_ind, ring_difference_per_segment)

    # 初始化 x2 数据
    SR_x2 = np.ndarray((400, 160, 160))
    RDPS_x2, matrix_size_x2 = ring_(20)
    matrix_ind_x2 = matrix_size2ind(matrix_size_x2)
    SR_x2_marked = SRMarking(SR=SR_x2,
                             matrix_ind=matrix_ind_x2,
                             ring_difference_per_segment=RDPS_x2, num_rings=20)

    SR_x2 = singram_resize(SR_marked, SR_x2_marked)

    # 初始化 x4 数据
    SR_x4 = np.ndarray((1600, 320, 320))
    RDPS_x4, matrix_size_x4 = ring_(40)
    matrix_ind_x4 = matrix_size2ind(matrix_size_x4)
    SR_x4_marked = SRMarking(SR=SR_x4,
                             matrix_ind=matrix_ind_x4,
                             ring_difference_per_segment=RDPS_x4, num_rings=40)

    SR_x4 = singram_resize(SR_x2, SR_x4_marked)

    return SR_x4


def x2TOx4(SR_x2):
    # 给原始数据（x2）做标记
    ring_difference_per_segment, matrix_size = ring_(20)
    matrix_ind = matrix_size2ind(matrix_size)
    SR_x2_marked = SRMarking(SR_x2, matrix_ind, ring_difference_per_segment, num_rings=20)

    # 初始化 x4 数据
    SR_x4 = np.ndarray((1600, 320, 320))
    RDPS_x4, matrix_size_x4 = ring_(40)
    matrix_ind_x4 = matrix_size2ind(matrix_size_x4)
    SR_x4_marked = SRMarking(SR=SR_x4,
                             matrix_ind=matrix_ind_x4,
                             ring_difference_per_segment=RDPS_x4, num_rings=40)

    SR_x4 = singram_resize(SR_x2_marked, SR_x4_marked)

    return SR_x4


def x1_to_x4(work_dir):
    S = np.fromfile(str(work_dir/'sinogram.s'), dtype=np.float32)
    SR = S.reshape(100, 80, 80)

    SR_x1TOx4 = x1TOx4(SR)
    (np.array(list(SR_x1TOx4.values()), dtype=np.float32)).tofile(str(work_dir/'sinogram_x1TOx4.s'))


def x2_to_x4(work_dir):
    S = np.fromfile(str(work_dir/'sinogram.s'), dtype=np.float32)
    SR_x2 = S.reshape(400, 160, 160)

    SR_x2TOx4 = x2TOx4(SR_x2)
    (np.array(list(SR_x2TOx4.values()), dtype=np.float32)).tofile(str(work_dir/'sinogram_x2TOx4.s'))


def run_recon(sub_workdir):
    print("    Run root2bin")
    subprocess.run(["bash", "root2bintxt.sh"], cwd=str(sub_workdir))

    print("    Run txt2h5")
    subprocess.run(["python", "txt2h5.py"], cwd=str(sub_workdir))

    print("    Run run_stir")
    subprocess.run(["bash", "run_stir.sh"], cwd=str(sub_workdir))