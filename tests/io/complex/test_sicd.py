import os
import time

# from . import unittest

from sarpy.io.complex.sicd import SICDDetails

# TODO: refactor into proper unit tests


def sicd_headers():
    test_root = os.path.expanduser(os.path.join('~', 'Desktop', 'sarpy_testing', 'sicd'))
    for fil in [
        'sicd_example_RMA_RGZERO_RE16I_IM16I.nitf',
        # 'sicd_example_RMA_RGZERO_RE32F_IM32F.nitf',
        # 'sicd_example_RMA_RGZERO_RE32F_IM32F_cropped_multiple_image_segments_v1.2.nitf',
    ]:
        test_file = os.path.join(test_root, fil)

        start = time.time()
        details = SICDDetails(test_file)
        # how long does it take to unpack details?
        print('unpacked sicd details in {}'.format(time.time() - start))

        # does it register as a sicd?
        assert details.is_sicd

        # how does the sicd_meta look?
        print(details.sicd_meta)

        # how do the image headers look?
        for i, entry in enumerate(details.img_headers):
            print('image header {}\n{}'.format(i, entry))


if __name__ == '__main__':
    sicd_headers()
