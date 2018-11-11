from jfs.api import Directory as DIR
import pathlib

class Directory(DIR):
    def __init__(self, path, load_depth=0):
        super().__init__(path, load_depth)

    @property
    def pathlib_dir(self):
        return pathlib.Path(self.path.abs)

    def list_subdir(self):
        return [x for x in self.pathlib_dir.iterdir() if x.is_dir()]