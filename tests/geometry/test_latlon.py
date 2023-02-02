#
# Copyright 2022 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.geometry import latlon


def test_string():
    ll_str = latlon.string(33.92527777777778, 'lat')  # float input
    assert ll_str == '33°55\'31"N'

    ll_str = latlon.string(33, 'lat')
    assert ll_str == '33°00\'00"N'

    ll_str = latlon.string(33.92527777777778, 'lat', num_units=1)
    assert ll_str == '33.92528°N'
    ll_str = latlon.string(33.92527777777778, 'lat', num_units=2)
    assert ll_str == "33°56'N"
    ll_str = latlon.string(33.92527777777778, 'lat', num_units=3)
    assert ll_str == '33°55\'31"N'

    ll_str = latlon.string(33.92527777777778, 'lat', include_symbols=False)
    assert ll_str == '335531N'

    ll_str = latlon.string(33.92527777777778, 'lat', signed=True)
    assert ll_str == '+33°55\'31"'

    ll_str = latlon.string(-33.92527777777778, 'lat')
    assert ll_str == '33°55\'31"S'

    ll_str = latlon.string(np.array([33.0, 55.0, 31.0]), 'lat')  # array input
    assert ll_str == '33°55\'31"N'
    ll_str = latlon.string([33.0, 55.0, 31.0], 'lat')  # list input
    assert ll_str == '33°55\'31"N'
    ll_str = latlon.string((33.0, 55.0, 31.0), 'lat')  # tuple input
    assert ll_str == '33°55\'31"N'

    ll_str = latlon.string(133.92527777777778, 'lon')
    assert ll_str == '133°55\'31"E'
    ll_str = latlon.string(493.92527777777778, 'lon')
    assert ll_str == '133°55\'31"E'
    ll_str = latlon.string(-133.92527777777778, 'lon')
    assert ll_str == '133°55\'31"W'

    ll_str = latlon.string([33.0, 55.0, 60.0], 'lat')  # seconds == 60
    assert ll_str == '33°56\'00"N'

    ll_str = latlon.string([33.0, 59.0, 60.0], 'lat')  # seconds rollover to minutes == 60
    assert ll_str == '34°00\'00"N'

    ll_str = latlon.string(33.0, 'lat', padded=False)
    assert ll_str == '33°0\'0"N'


def test_dms():
    deg, min, sec = latlon.dms(33.92527777777778)
    assert deg == 33
    assert min == 55
    assert 31 == pytest.approx(sec, abs=1e-10)

    deg, min, sec = latlon.dms(-33.92527777777778)
    assert deg == -33
    assert min == 55
    assert 31 == pytest.approx(sec, abs=1e-10)


def test_num():
    ll_dec = latlon.num('33:55:31')
    assert 33.9252777778 == pytest.approx(ll_dec, abs=1e-10)

    ll_dec = latlon.num('33:55:31W')
    assert -33.9252777778 == pytest.approx(ll_dec, abs=1e-10)

    ll_dec = latlon.num('1335531')
    assert 133.9252777778 == pytest.approx(ll_dec, abs=1e-10)

    ll_dec = latlon.num('360:00:01')
    assert np.isnan(ll_dec)
    ll_dec = latlon.num('-180:00:01')
    assert np.isnan(ll_dec)
    ll_dec = latlon.num('33:55:31:21')
    assert np.isnan(ll_dec)

    with pytest.raises(ValueError):
        ll_dec = latlon.num(33.92527777777778)
