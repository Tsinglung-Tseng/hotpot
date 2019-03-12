from srf.io.listmode import save_h5
import numpy as np

data = np.fromfile("normal_scan_true.txt", dtype=np.float32).reshape(-1,7)
result = {'fst': data[:, :3], 'snd': data[:, 3:6], 'weight': np.ones_like(data[:,0])}
save_h5('input.h5', result)