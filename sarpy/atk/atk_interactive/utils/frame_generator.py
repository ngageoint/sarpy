from algorithm_toolkit.atk import app
from PIL import Image
import io
import os
import imageio

from sarpy.atk.atk_interactive.utils import atk_tools


class FrameGenerator(object):
    def __init__(self):
        self.decimation = 10

        img_path = os.path.join(os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__))), "resources/logo.jpeg")

        self.sarpy_reader = None
        self.numpy_data = [imageio.imread(img_path)]
        self.atk_chains = atk_tools.AtkChains()

    def set_image_path(self, pth):

        chain_name = 'open_nitf'

        chain_json = self.atk_chains.get_chain_json(chain_name)
        chain_json['algorithms'][0]['parameters']['filename'] = pth
        self.atk_chains.set_chain_json(chain_name, chain_json)

        status_key = atk_tools.call_atk_chain(self.atk_chains, chain_name)

        ch = app.config['CHAIN_HISTORY']

        self.sarpy_reader = ch[status_key].metadata['sarpy_reader']
        self.update_frame()

        nx = self.sarpy_reader.sicdmeta.ImageData.NumCols
        ny = self.sarpy_reader.sicdmeta.ImageData.NumRows
        return nx, ny

    def set_decimation(self, dec):
        self.decimation = dec
        self.update_frame()

    def crop_image(self, xmin, ymin, xmax, ymax):
        # TODO update globals
        #  (minx, miny, maxx, maxy)
        bounds = [xmin, ymin, xmax, ymax]
        self.update_frame(bounds=bounds)

    def ortho_image(self, output_path):

        ro = self.sarpy_reader
        dec = self.decimation
        pix = self.numpy_data[0]
        chain_name = 'save_ortho'

        chain_json = self.atk_chains.get_chain_json(chain_name)
        chain_json['algorithms'][0]['parameters']['sarpy_reader'] = ro
        chain_json['algorithms'][0]['parameters']['decimation'] = dec
        chain_json['algorithms'][0]['parameters']['remapped_data'] = pix
        chain_json['algorithms'][0]['parameters']['geotiff_path'] = output_path

        self.atk_chains.set_chain_json(chain_name, chain_json)

        atk_tools.call_atk_chain(self.atk_chains, chain_name, pass_params_in_mem=True)

        print("Ortho creation completed!")

        return ''

    def update_frame(self, bounds=None):
        chain_name = 'remap_data'

        ro = self.sarpy_reader
        dec = self.decimation

        chain_json = self.atk_chains.get_chain_json(chain_name)
        chain_json['algorithms'][0]['parameters']['sarpy_reader'] = ro
        chain_json['algorithms'][0]['parameters']['decimation'] = dec

        if bounds is not None:
            chain_json['algorithms'][0]['parameters']['ystart'] = bounds[1]
            chain_json['algorithms'][0]['parameters']['yend'] = bounds[3]
            chain_json['algorithms'][0]['parameters']['xstart'] = bounds[0]
            chain_json['algorithms'][0]['parameters']['xend'] = bounds[2]

        self.atk_chains.set_chain_json(chain_name, chain_json)

        status_key = atk_tools.call_atk_chain(self.atk_chains, chain_name, pass_params_in_mem=True)

        ch = app.config['CHAIN_HISTORY']
        pix = ch[status_key].metadata['remapped_data']

        self.numpy_data = [pix]

    def get_frame(self):

        img = Image.fromarray(self.numpy_data[0].astype('uint8'))  # convert arr to image

        file_object = io.BytesIO()  # create file in memory
        img.save(file_object, format='png')  # save as jpg in file in memory
        file_object.seek(0)  # move to beginning of file

        png_data = file_object.read()

        return png_data
