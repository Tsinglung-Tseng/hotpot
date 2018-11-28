from .base import sin, cos, tan


class Config:
    P_SIZE = 512
    R = 210
    N = 20

    @property
    def rs(self):
        return [R / (4 * (i + tan(60))) for i in range(self.N)]
