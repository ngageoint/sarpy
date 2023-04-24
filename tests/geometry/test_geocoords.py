import pytest

import numpy

from sarpy.geometry import geocoords


EQUATORIAL_RADIUS = 6378137
POLAR_RADIUS = 6356752.314245179
TOLERANCE = 1e-8

numpy.random.seed(314159)


@pytest.fixture(scope='module')
def input():
    llh = numpy.array([[0, 0, 0], [0, 180, 0], [90, 0, 0], [-90, 0, 0], [0, 90, 0]], dtype='float64')
    ecf = numpy.array([[EQUATORIAL_RADIUS, 0, 0],
                       [-EQUATORIAL_RADIUS, 0, 0],
                       [0, 0, POLAR_RADIUS],
                       [0, 0, -POLAR_RADIUS],
                       [0, EQUATORIAL_RADIUS, 0]], dtype='float64')
    ned = numpy.array([[0,  0,  0],
                       [0,  0,  EQUATORIAL_RADIUS*2],
                       [POLAR_RADIUS,  0,  EQUATORIAL_RADIUS],
                       [-POLAR_RADIUS,  0,  EQUATORIAL_RADIUS],
                       [0,  EQUATORIAL_RADIUS,  EQUATORIAL_RADIUS]], dtype='float64')
    enu = numpy.array([[0,  0,  0],
                       [0,  0,  -EQUATORIAL_RADIUS*2],
                       [0,  POLAR_RADIUS,  -EQUATORIAL_RADIUS],
                       [0,  -POLAR_RADIUS,  -EQUATORIAL_RADIUS],
                       [EQUATORIAL_RADIUS, 0,  -EQUATORIAL_RADIUS]], dtype='float64')
    orp = ecf[0, :]

    return {"llh": llh, "ecf": ecf, "ned": ned, "enu": enu, "orp": orp}


def test_ecf_to_geodetic(input):
    out = geocoords.ecf_to_geodetic(input['ecf'][0, :])
    # basic value check
    assert out == pytest.approx(input['llh'][0, :], abs=TOLERANCE)

    out2 = geocoords.ecf_to_geodetic(input['ecf'])
    # 2d value check
    assert out2 == pytest.approx(input['llh'], abs=TOLERANCE)

    # check (lon, lat, hae) order
    out3 = geocoords.ecf_to_geodetic(input['ecf'], ordering='longlat')
    assert out3 == pytest.approx(input['llh'][:, [1, 0, 2]], abs=TOLERANCE)

    # error check
    with pytest.raises(ValueError):
        geocoords.ecf_to_geodetic(numpy.arange(4))


def test_geodetic_to_ecf(input):
    out = geocoords.geodetic_to_ecf(input['llh'][0, :])
    # basic value check
    assert out == pytest.approx(input['ecf'][0, :], abs=TOLERANCE)

    out2 = geocoords.geodetic_to_ecf(input['llh'])
    assert out2 == pytest.approx(input['ecf'], abs=TOLERANCE)

    # check (lon, lat, hae) order
    out3 = geocoords.geodetic_to_ecf(input['llh'][:, [1, 0, 2]], ordering='longlat')
    assert out3 == pytest.approx(input['ecf'], abs=TOLERANCE)

    # error check
    with pytest.raises(ValueError):
        geocoords.geodetic_to_ecf(numpy.arange(4))


def test_values_both_ways():
    shp = (8, 5)
    rand_llh = numpy.empty(shp + (3, ), dtype=numpy.float64)
    rng = numpy.random.default_rng()
    rand_llh[:, :, 0] = 180*(rng.random(shp) - 0.5)
    rand_llh[:, :, 1] = 360*(rng.random(shp) - 0.5)
    rand_llh[:, :, 2] = 1e5*rng.random(shp)

    rand_ecf = geocoords.geodetic_to_ecf(rand_llh)
    rand_llh2 = geocoords.ecf_to_geodetic(rand_ecf)
    rand_ecf2 = geocoords.geodetic_to_ecf(rand_llh2)

    # llh match
    assert rand_llh == pytest.approx(rand_llh2, abs=TOLERANCE)

    # ecf match
    assert rand_ecf == pytest.approx(rand_ecf2, abs=TOLERANCE)


def test_ecf_to_ned(input):
    out = geocoords.ecf_to_ned(input['ecf'][0, :], input['orp'])
    assert numpy.all(out == 0)

    out = geocoords.ecf_to_ned(input['ecf'], input['orp'])
    assert out == pytest.approx(input['ned'], abs=TOLERANCE)

    # orp is a list
    out = geocoords.ecf_to_ned(input['ecf'], [EQUATORIAL_RADIUS, 0, 0])
    assert out == pytest.approx(input['ned'], abs=TOLERANCE)

    # absolute_coords not default
    out = geocoords.ecf_to_ned(input['ecf'][0, :], input['orp'], absolute_coords=False)
    assert out == pytest.approx([0, 0, -EQUATORIAL_RADIUS], abs=TOLERANCE)

    # orp is of length 3
    with pytest.raises(ValueError):
        orp1 = numpy.append(input['orp'], 0)
        out = geocoords.ecf_to_ned(input['ecf'][0, :], orp1)


def test_ned_to_ecf(input):
    # input is list instead of array
    out = geocoords.ned_to_ecf(input['ned'][0, :].tolist(), input['orp'])
    assert out == pytest.approx(input['ecf'][0, :], abs=TOLERANCE)

    out = geocoords.ned_to_ecf(input['ned'], input['orp'])
    assert out == pytest.approx(input['ecf'], abs=TOLERANCE)

    # orp is a list
    out = geocoords.ned_to_ecf(input['ned'], [EQUATORIAL_RADIUS, 0, 0])
    assert out == pytest.approx(input['ecf'], abs=TOLERANCE)


def test_ecf_to_ned_roundtrip(input):
    shp = (8, 5)
    rand_ecf = numpy.empty(shp + (3, ), dtype=numpy.float64)
    rng = numpy.random.default_rng()
    rand_ecf[:, :, 0] = EQUATORIAL_RADIUS*(rng.random(shp) - 0.5)
    rand_ecf[:, :, 1] = EQUATORIAL_RADIUS*(rng.random(shp) - 0.5)
    rand_ecf[:, :, 2] = POLAR_RADIUS*(rng.random(shp) - 0.5)

    rand_ned = geocoords.ecf_to_ned(rand_ecf, input['orp'])
    rand_ecf2 = geocoords.ned_to_ecf(rand_ned, input['orp'])
    rand_ned2 = geocoords.ecf_to_ned(rand_ecf2, input['orp'])

    # ecf match
    assert rand_ecf == pytest.approx(rand_ecf2, abs=TOLERANCE)

    # ned match
    assert rand_ned == pytest.approx(rand_ned2, abs=TOLERANCE)


def test_ecf_to_enu(input):
    out = geocoords.ecf_to_enu(input['ecf'][0, :], input['orp'])
    assert numpy.all(out == 0)

    out = geocoords.ecf_to_enu(input['ecf'], input['orp'])
    assert out == pytest.approx(input['enu'], abs=TOLERANCE)

    # orp is a list
    out = geocoords.ecf_to_enu(input['ecf'], [EQUATORIAL_RADIUS, 0, 0])
    assert out == pytest.approx(input['enu'], abs=TOLERANCE)

    # absolute_coords not default
    out = geocoords.ecf_to_enu(input['ecf'][0, :], input['orp'], absolute_coords=False)
    assert out == pytest.approx([0, 0, EQUATORIAL_RADIUS], abs=TOLERANCE)

    # orp is of length 3
    with pytest.raises(ValueError):
        orp1 = numpy.append(input['orp'], 0)
        out = geocoords.ecf_to_enu(input['ecf'][0, :], orp1)


def test_enu_to_ecf(input):
    out = geocoords.enu_to_ecf(input['enu'][0, :], input['orp'])
    assert out == pytest.approx(input['ecf'][0, :], abs=TOLERANCE)

    out = geocoords.enu_to_ecf(input['enu'], input['orp'])
    assert out == pytest.approx(input['ecf'], abs=TOLERANCE)

    # orp is a list
    out = geocoords.enu_to_ecf(input['enu'], [EQUATORIAL_RADIUS, 0, 0])
    assert out == pytest.approx(input['ecf'], abs=TOLERANCE)


def test_ecf_to_enu_roundtrip(input):
    shp = (8, 5)
    rand_ecf = numpy.empty(shp + (3, ), dtype=numpy.float64)
    rng = numpy.random.default_rng()
    rand_ecf[:, :, 0] = EQUATORIAL_RADIUS*(rng.random(shp) - 0.5)
    rand_ecf[:, :, 1] = EQUATORIAL_RADIUS*(rng.random(shp) - 0.5)
    rand_ecf[:, :, 2] = POLAR_RADIUS*(rng.random(shp) - 0.5)

    rand_enu = geocoords.ecf_to_enu(rand_ecf, input['orp'])
    rand_ecf2 = geocoords.enu_to_ecf(rand_enu, input['orp'])
    rand_enu2 = geocoords.ecf_to_enu(rand_ecf2, input['orp'])

    # ecf match
    assert rand_ecf == pytest.approx(rand_ecf2, abs=TOLERANCE)

    # enu match
    assert rand_enu == pytest.approx(rand_enu2, abs=TOLERANCE)


def test_wgs84_norm(input):
    wgs84_norm = geocoords.wgs_84_norm(input['ecf'])
    expected = numpy.array([[1.,  0.,  0.],
                            [-1.,  0.,  0.],
                            [0.,  0.,  1.],
                            [0.,  0., -1.],
                            [0.,  1.,  0.]])
    assert wgs84_norm == pytest.approx(expected, abs=TOLERANCE)
