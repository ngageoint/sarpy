from algorithm_toolkit.atk import app
from PIL import Image
import io
import os
import imageio

import sarpy.visualization.remap as remap
from sarpy.atk.atk_interactive.utils import atk_tools


class FrameGenerator(object):
    def __init__(self):
        self.decimation = 10

        img_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "resources/logo.jpeg")

        self.sarpy_reader = None
        self.numpy_data = [imageio.imread(img_path)]

    def set_image_path(self, pth):
        chain_filename = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "chain_forms/open_nitf.json")

        chain_json = atk_tools.get_atk_form_json(chain_filename)
        chain_json['algorithms'][0]['parameters']['filename'] = pth

        atk_tools.call_atk_chain(chain_json, 'secret')

        cl = app.config['CHAIN_HISTORY']

        self.sarpy_reader = cl[-1].metadata['sarpy_reader']
        self.update_frame()

    def set_decimation(self, dec):
        self.decimation = dec
        self.update_frame()

    def update_frame(self):
        ro = self.sarpy_reader
        cdata = ro.read_chip[::self.decimation, ::self.decimation]
        self.numpy_data = [remap.density(cdata)]

    def get_frame(self):

        img = Image.fromarray(self.numpy_data[0].astype('uint8'))  # convert arr to image

        file_object = io.BytesIO()  # create file in memory
        img.save(file_object, format='png')  # save as jpg in file in memory
        file_object.seek(0)  # move to beginning of file

        png_data = file_object.read()

        return png_data
