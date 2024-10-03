#
# Copyright 2024 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import pytest

import sarpy.io.general.nitf_elements.image


def __make_band_bytes(numbands):
    banddef = {
        'NBANDS': f'{numbands}'.encode() if numbands < 10 else b'0',
        'XBANDS': b'' if numbands < 10 else f'{numbands:05d}'.encode(),
    }
    for n in range(numbands):
        banddef.update({
            f'IREPBAND{n:06d}': b'  ',
            f'ISUBCAT{n:06d}': f'cat{n:03d}'.encode(),
            f'IFC{n:06d}': b'N',
            f'IMFLT{n:06d}': b'   ',
            f'NLUTS{n:06d}': b'0',
        })
    return b''.join(banddef.values())


def test_imagebands_minlength():
    assert sarpy.io.general.nitf_elements.image.ImageBands.minimum_length() == len(__make_band_bytes(1))


@pytest.mark.parametrize('num_bands', (1, 24))
def test_imagebands(num_bands):
    band_bytes = __make_band_bytes(num_bands)
    parsed_bands = sarpy.io.general.nitf_elements.image.ImageBands.from_bytes(band_bytes, 0)
    assert len(parsed_bands.values) == num_bands
    assert all(int(x.ISUBCAT[3:]) == n for n, x in enumerate(parsed_bands.values))
    assert parsed_bands.get_bytes_length() == len(band_bytes)
    assert parsed_bands.to_bytes() == band_bytes


def test_imagebands_values_update():
    parsed_bands = sarpy.io.general.nitf_elements.image.ImageBands.from_bytes(__make_band_bytes(1), 0)
    for nbands in range(1, 11):
        parsed_bands.values = [parsed_bands.values[0]] * nbands
        orig_bytes = parsed_bands.to_bytes()
        round_trip_bytes = sarpy.io.general.nitf_elements.image.ImageBands.from_bytes(orig_bytes, 0).to_bytes()
        assert orig_bytes == round_trip_bytes
