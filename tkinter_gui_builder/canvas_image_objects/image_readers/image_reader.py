import numpy


class AbstractImageReader:
    fname = str
    full_image_nx = int
    full_image_ny = int

    def init_w_fname(self,
                     fname,  # type: str
                     ):         # type: (...) -> AbstractImageReader
        raise NotImplementedError

    def get_image_chip(self,
                       y_start,     # type: int
                       y_end,       # type: int
                       x_start,     # type: int
                       x_end,       # type: int
                       decimation,  # type: int
                       ):           # type: (...) -> numpy.ndarray
        raise NotImplementedError
