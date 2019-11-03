from algorithm_toolkit import Algorithm, AlgorithmChain
import base64
from PIL import Image
import io


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        pix = params['numpy_image_data']
        img = Image.fromarray(pix.astype('uint8'))

        file_object = io.BytesIO()
        img.save(file_object, format='png')
        file_object.seek(0)

        img_bytes = base64.b64encode(file_object.read())
        img_str = img_bytes.decode('utf-8')

        width, height = img.size

        chain_output = {
            "output_type": "geo_raster",
            "output_value": {
                "extent": [0,0,height,width],
                "raster": img_str
            }
        }
        cl.add_to_metadata('chain_output_value', chain_output)

        # Do not edit below this line
        return cl
