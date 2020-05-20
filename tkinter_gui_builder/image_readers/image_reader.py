import numpy


# TODO add functionality for multiple bands
class ImageReader:
    fname = str
    full_image_nx = int
    full_image_ny = int

    def __getitem__(self, key):
        raise NotImplementedError
