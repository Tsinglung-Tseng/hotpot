from jfs.api import Path, File, mkdir, mv, cp, rm
from .base import Directory
from pygate.routine.base import Operation, OperationOnFile, OperationOnSubdirectories, OpeartionWithShellCall, RoutineOnDirectory
from typing import Iterable
import subprocess


class KEYS:
    SUBDIRECTORIES = 'subdirectories'
    TARGET = 'target'
    IS_TO_BROADCAST = 'is_to_broadcast'
    CONTENT = 'content'
    TO_BROADCAST_FILES = 'to_broadcast_files'


class OpSubdirectoriesMaker(Operation):
    def __init__(self, nb_split: int, subdirectory_format: str="sub.{}"):
        self.nb_split = nb_split
        self.fmt = subdirectory_format

    def apply(self, r: RoutineOnDirectory):
        result = self.dryrun(r)
        for n in result[KEYS.SUBDIRECTORIES]:
            mkdir(Directory(r.directory.path.abs+"/"+n))
        return result

    def dryrun(self, r: RoutineOnDirectory):
        return {KEYS.SUBDIRECTORIES: tuple([self.fmt.format(i) for i in range(self.nb_split)])}


class OpCmdOnSubdir(OperationOnSubdirectories):
    def __init__(self, patterns: Iterable[str], command: Iterable[str]):
        super().__init__(patterns)
        self.command = command

    def apply(self, r: RoutineOnDirectory):
        return self.subdirectories(r).to_list().to_blocking().first()


if __name__=="__main__":
    d = Directory('/Users/tsinglung/subdir.test')
    o = OpSubdirectoriesMaker(11)
    r = RoutineOnDirectory(d, [o], dryrun=False)
    # o.apply(r)
    # r.work()

    cosd = OpCmdOnSubdir(patterns=["sub.{}"], command=["python", "init", "subdir", "-n", "10"])
    r = RoutineOnDirectory(d, [cosd], dryrun=False)


