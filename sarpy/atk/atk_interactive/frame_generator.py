from algorithm_toolkit.atk import app
from PIL import Image
import io
import sarpy.io.complex as cf
import sarpy.visualization.remap as remap
import os


class FrameGenerator(object):
    def __init__(self):
        self.decimation = 10

        fname = '~/.sarpy/support_data/nitf/sicd_example_1_PFA_RE32F_IM32F_HH.nitf'
        if fname.startswith('~'):
            fname = os.path.expanduser(fname)

        ro = cf.open(fname)
        app.config['sarpy_reader'] = ro

        cdata = ro.read_chip[::self.decimation, ::self.decimation]
        self.numpy_data = [remap.density(cdata)]

    def set_decimation(self, dec):
        self.decimation = dec
        self.update_frame()

    def update_frame(self):
        ro = app.config['sarpy_reader']
        cdata = ro.read_chip[::self.decimation, ::self.decimation]
        self.numpy_data = [remap.density(cdata)]

    def get_frame(self):

        img = Image.fromarray(self.numpy_data[0].astype('uint8'))  # convert arr to image

        file_object = io.BytesIO()  # create file in memory
        img.save(file_object, format='png')  # save as jpg in file in memory
        file_object.seek(0)  # move to beginning of file

        png_data = file_object.read()

        return png_data
