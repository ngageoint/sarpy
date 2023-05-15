#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import ImageData


def test_imagedata(sicd, kwargs, caplog):
    image_type = ImageData.FullImageType()
    assert image_type.NumRows is None
    assert image_type.NumCols is None

    image_type = image_type.from_array([sicd.ImageData.NumRows, sicd.ImageData.NumCols])
    assert image_type.NumRows == sicd.ImageData.NumRows
    assert image_type.NumCols == sicd.ImageData.NumCols

    with pytest.raises(ValueError, match='Expected array to be of length 2, and received 1'):
        image_type.from_array([sicd.ImageData.NumRows])
    with pytest.raises(ValueError, match='Expected array to be numpy.ndarray, list, or tuple'):
        image_type.from_array(image_type)

    image_type1 = ImageData.FullImageType(sicd.ImageData.NumRows, sicd.ImageData.NumCols, **kwargs)
    assert image_type1._xml_ns == kwargs['_xml_ns']
    assert image_type1._xml_ns_key == kwargs['_xml_ns_key']
    assert image_type1.NumRows == sicd.ImageData.NumRows
    assert image_type1.NumCols == sicd.ImageData.NumCols

    image_array = image_type.get_array()
    assert np.all(image_array == np.array([sicd.ImageData.NumRows, sicd.ImageData.NumCols]))

    amp_table = np.ones((256, 256))
    image_data = ImageData.ImageDataType('AMP8I_PHS8I',
                                         None,
                                         sicd.ImageData.NumRows,
                                         sicd.ImageData.NumCols,
                                         sicd.ImageData.FirstRow,
                                         sicd.ImageData.FirstCol,
                                         sicd.ImageData.FullImage,
                                         sicd.ImageData.SCPPixel,
                                         sicd.ImageData.ValidData,
                                         **kwargs)
    assert image_data._xml_ns == kwargs['_xml_ns']
    assert image_data._xml_ns_key == kwargs['_xml_ns_key']
    assert image_data.get_pixel_size() == 2
    assert not image_data._basic_validity_check()
    assert "We have `PixelType='AMP8I_PHS8I'` and `AmpTable` is not defined for ImageDataType" in caplog.text

    image_data = ImageData.ImageDataType('RE32F_IM32F',
                                         amp_table,
                                         sicd.ImageData.NumRows,
                                         sicd.ImageData.NumCols,
                                         sicd.ImageData.FirstRow,
                                         sicd.ImageData.FirstCol,
                                         sicd.ImageData.FullImage,
                                         sicd.ImageData.SCPPixel,
                                         sicd.ImageData.ValidData,
                                         **kwargs)
    assert image_data.get_pixel_size() == 8
    assert not image_data._basic_validity_check()
    assert "We have `PixelType != 'AMP8I_PHS8I'` and `AmpTable` is defined for ImageDataType" in caplog.text

    image_data = ImageData.ImageDataType('RE32F_IM32F',
                                         None,
                                         sicd.ImageData.NumRows,
                                         sicd.ImageData.NumCols,
                                         sicd.ImageData.FirstRow,
                                         sicd.ImageData.FirstCol,
                                         sicd.ImageData.FullImage,
                                         sicd.ImageData.SCPPixel,
                                         sicd.ImageData.ValidData,
                                         **kwargs)
    assert image_data._basic_validity_check()

    assert image_data._check_valid_data()

    valid_vertex_data = image_data.get_valid_vertex_data()
    assert len(valid_vertex_data) == len(sicd.ImageData.ValidData)

    full_vertex_data = image_data.get_full_vertex_data()
    assert np.all(full_vertex_data == np.array([[0, 0],
                                                [0, image_data.NumCols - 1],
                                                [image_data.NumRows - 1, image_data.NumCols - 1],
                                                [image_data.NumRows - 1, 0]]))

    image_data.PixelType = 'RE16I_IM16I'
    assert image_data.get_pixel_size() == 4
