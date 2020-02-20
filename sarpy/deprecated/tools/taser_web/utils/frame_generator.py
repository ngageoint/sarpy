from algorithm_toolkit.atk import app
from PIL import Image
import io
import os
import imageio
import base64

from sarpy.deprecated.tools.atk_utils import atk_tools


class FrameGenerator(object):
    def __init__(self):
        self.decimation_x = None
        self.decimation_y = None
        self.decimation_curr = None
        self.bounds = None

        self.nx = None
        self.ny = None

        img_path = os.path.join(os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__))), "resources/logo.jpeg")

        self.sarpy_reader = None
        self.numpy_data = [imageio.imread(img_path)]
        self.atk_chains = atk_tools.AtkChains(project_name="taser_web")

    def check_decimation(self):
        if self.decimation_x < 1:
            self.decimation_x = 1
        if self.decimation_y < 1:
            self.decimation_y = 1

    def set_image_path(self, pth, tnx, tny):

        chain_name = 'open_nitf'

        chain_json = self.atk_chains.get_chain_json(chain_name)
        chain_json['algorithms'][0]['parameters']['filename'] = pth
        self.atk_chains.set_chain_json(chain_name, chain_json)

        status_key, _ = atk_tools.call_atk_chain(self.atk_chains, chain_name)

        ch = app.config['CHAIN_HISTORY']

        self.sarpy_reader = ch[status_key].metadata['sarpy_reader']
        self.nx = self.sarpy_reader.sicdmeta.ImageData.NumCols
        self.ny = self.sarpy_reader.sicdmeta.ImageData.NumRows

        self.decimation_x = int(round(float(self.nx)/float(tnx)))
        self.decimation_y = int(round(float(self.ny)/float(tny)))

        self.check_decimation()

        ul = [0, 0]  # y, x
        lr = [self.ny, self.nx]
        self.bounds = [ul, lr]

        self.update_frame()

        return self.nx, self.ny

    def crop_image(self, xmin, ymin, xmax, ymax, tnx, tny):
        self.decimation_x = int(round(float(xmax-xmin)/float(tnx), 0))
        self.decimation_y = int(round(float(ymax-ymin)/float(tny), 0))
        self.check_decimation()

        bounds = [xmin, ymin, xmax, ymax]
        self.update_frame(bounds=bounds)

    def ortho_image(self, output_path):

        ro = self.sarpy_reader
        # TODO Update to use x and y
        dec = self.decimation_x
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

        if self.decimation_x > self.decimation_y:
            dec = self.decimation_x
        else:
            dec = self.decimation_y

        self.decimation_curr = dec

        chain_json = self.atk_chains.get_chain_json(chain_name)
        chain_json['algorithms'][0]['parameters']['sarpy_reader'] = ro
        chain_json['algorithms'][0]['parameters']['decimation'] = dec

        if bounds is not None:
            chain_json['algorithms'][0]['parameters']['ystart'] = bounds[1]
            chain_json['algorithms'][0]['parameters']['yend'] = bounds[3]
            chain_json['algorithms'][0]['parameters']['xstart'] = bounds[0]
            chain_json['algorithms'][0]['parameters']['xend'] = bounds[2]

            ul = [bounds[1], bounds[0]]  # y, x
            lr = [bounds[3], bounds[2]]
            self.bounds = [ul, lr]

        self.atk_chains.set_chain_json(chain_name, chain_json)

        status_key, _ = atk_tools.call_atk_chain(self.atk_chains, chain_name, pass_params_in_mem=True)

        ch = app.config['CHAIN_HISTORY']
        pix = ch[status_key].metadata['remapped_data']

        self.numpy_data = [pix]

    def get_frame(self):

        img = Image.fromarray(self.numpy_data[0].astype('uint8'))

        file_object = io.BytesIO()
        img.save(file_object, format='png')
        file_object.seek(0)

        img_bytes = base64.b64encode(file_object.read())
        img_str = img_bytes.decode('utf-8')

        chain_output = {
            "output_type": "geo_raster",
            "output_value": {
                "extent": str(self.bounds),
                "raster": img_str,
                "decimation": self.decimation_curr
            }
        }

        return chain_output
