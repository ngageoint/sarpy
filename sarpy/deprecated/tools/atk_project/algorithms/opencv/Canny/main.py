from algorithm_toolkit import Algorithm, AlgorithmChain

import cv2 as cv


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here
        img = params['image_array']

        threshold1 = params['threshold1']
        threshold2 = params['threshold2']

        apertureSize = 3
        if 'apertureSize' in params:
            apertureSize = params['apertureSize']

        l2grad = False
        if 'L2gradient' in params:
            l2grad = params['L2gradient'].lower() == 'true'

        try:
            output = cv.Canny(img, threshold1, threshold2,
                              apertureSize=apertureSize, L2gradient=l2grad)
        except Exception as e:
            self.raise_client_error(str(e))

        cl.add_to_metadata('image_array', output)
        # Do not edit below this line
        return cl
