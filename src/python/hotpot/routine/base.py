from jfs.api import Directory as DIR
from pygate.routine.base import Operation, RoutineOnDirectory
from typing import Iterable
import pathlib

class Directory(DIR):
    def __init__(self, path, load_depth=0):
        super().__init__(path, load_depth)

    @property
    def pathlib_dir(self):
        return pathlib.Path(self.path.abs)

    def list_subdir(self):
        return [x for x in self.pathlib_dir.iterdir() if x.is_dir()]


class OperationOnSubdirectories(Operation):
    def __init__(self, patterns: Iterable[str]):
        self.patterns = patterns

    def subdirectories(self, r: RoutineOnDirectory) -> 'Observable[Directory]':
        from dxl.fs import match_directory
        return (r.directory.listdir_as_observable()
                .filter(match_directory(self.patterns)))