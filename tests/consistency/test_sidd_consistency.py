#
# Licensed under MIT License.  See LICENSE.
#
import numpy
import pytest
import tempfile

from pathlib import Path

import sarpy.io.product.sidd2_elements.SIDD as sarpy_sidd2
from sarpy.consistency.sidd_consistency import check_file
from sarpy.io.product.sidd import SIDDWriter


@pytest.fixture()
def sidd_meta():
    sidd_meta = sarpy_sidd2.SIDDType.from_xml_file(Path(__file__).parents[1] / 'data/example.sidd.xml')
    
    return sidd_meta


@pytest.fixture
def rgb24i_sidd(sidd_meta):
    sidd_meta.Display.PixelType = "RGB24I"
    sidd_meta.Display.NumBands = 3

    with tempfile.NamedTemporaryFile(delete=True) as sidd_file:
        writer = SIDDWriter(sidd_file, sidd_meta=sidd_meta)

        rows = sidd_meta.Measurement.PixelFootprint.Row
        cols = sidd_meta.Measurement.PixelFootprint.Row
        image = numpy.random.uniform(0, 1, size=(rows, cols, 3))
        image *= 255
        image = image.astype(numpy.uint8)
        writer(image, start_indices=(0, 0))
        writer.close()

        yield sidd_file.name


def test_rgb24I_sidd(rgb24i_sidd):
    assert check_file(rgb24i_sidd)
