from collections import UserDict


_default = {
    "data": '/home/qinglong/node3share/analytical_phantom_sinogram.h5',
    "summary_dir": '/home/qinglong/node3share/remote_drssrn/tensorboard_log/2xDown_new_1'
}

class Config(UserDict):
    def __init__(self):
        super(__class__, self).__init__()
        self.data.update(_default)


config = Config()