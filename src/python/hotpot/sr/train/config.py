from collections import UserDict


class Config(UserDict):
    def __init__(self, kwargs):
        super().__init__(kwargs)

    # def __getitem__(self, item):
    #     return UserDict.__getitem__(self, item) % self


config = {
    "data": '/home/qinglong/node3share/analytical_phantom_sinogram.h5',
    "graph": {
        "main_path": {
            "batch_size": 32,
            "down_sampling_rate": 2,
            "ROI_shape": (32, 64, 64, 1)
        }
    },
    "summary_dir": '/home/qinglong/node3share/remote_drssrn/tensorboard_log/2xDown_new_1'
}


c = Config(config)