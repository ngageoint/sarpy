import numpy


# TODO add functionality for multiple bands
class ImageReader:
    fname = str
    full_image_nx = int
    full_image_ny = int

    # TODO change this to slice notation, remove decimation argument (built in automatically)
    def get_image_chip(self,
                       y_start,     # type: int
                       y_end,       # type: int
                       x_start,     # type: int
                       x_end,       # type: int
                       decimation,  # type: int
                       ):           # type: (...) -> numpy.ndarray
        raise NotImplementedError
