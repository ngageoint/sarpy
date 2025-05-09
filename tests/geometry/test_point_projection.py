#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import pathlib

import numpy as np
import pytest

from sarpy.geometry import point_projection
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.product.sidd2_elements.SIDD import SIDDType
from sarpy.io.DEM.DEM import DEMInterpolator
TOLERANCE = 1e-8


@pytest.fixture(scope='module')
def sicd():
    xml_file = pathlib.Path(pathlib.Path.cwd(), 'tests/data/example.sicd.xml')
    structure = SICDType().from_xml_file(xml_file)
    scp_pixel = [structure.ImageData.SCPPixel.Row,
                 structure.ImageData.SCPPixel.Col]
    scp_ecf = [structure.GeoData.SCP.ECF.X,
               structure.GeoData.SCP.ECF.Y,
               structure.GeoData.SCP.ECF.Z]
    scp_llh = [structure.GeoData.SCP.LLH.Lat,
               structure.GeoData.SCP.LLH.Lon,
               structure.GeoData.SCP.LLH.HAE]

    return {'structure': structure, 'scp_pixel': scp_pixel, 'scp_ecf': scp_ecf, 'scp_llh': scp_llh}


@pytest.fixture(scope='module')
def sidd():
    xml_file = pathlib.Path(pathlib.Path.cwd(), 'tests/data/example.sidd.xml')
    return SIDDType().from_xml_file(xml_file)


def test_image_to_ground_plane(sicd):
    # project scp pixel (PLANE)
    scp_ecef1 = point_projection.image_to_ground(sicd['scp_pixel'],
                                                 sicd['structure'],
                                                 projection_type='PLANE')
    assert scp_ecef1 == pytest.approx(sicd['scp_ecf'], abs=TOLERANCE)

    scp_ecef2 = point_projection.image_to_ground([sicd['scp_pixel'], sicd['scp_pixel']],
                                                 sicd['structure'],
                                                 block_size=1,
                                                 projection_type='PLANE')
    assert np.all(np.abs(scp_ecef2[0] - scp_ecef1) < TOLERANCE)
    assert np.all(np.abs(scp_ecef2[1] - scp_ecef1) < TOLERANCE)

    # 2-dim gref
    gref = np.array([[sicd['structure'].GeoData.SCP.ECF.X],
                     [sicd['structure'].GeoData.SCP.ECF.Y],
                     [sicd['structure'].GeoData.SCP.ECF.Z]])
    scp_ecef3 = point_projection.image_to_ground(sicd['scp_pixel'],
                                                 sicd['structure'],
                                                 gref=gref,
                                                 projection_type='PLANE')
    assert scp_ecef3 == pytest.approx(sicd['scp_ecf'], abs=TOLERANCE)

    # 2-dim ugpn
    ugpn = gref
    scp_ecef4 = point_projection.image_to_ground(sicd['scp_pixel'],
                                                 sicd['structure'],
                                                 ugpn=ugpn,
                                                 projection_type='PLANE')
    assert scp_ecef4 == pytest.approx(sicd['scp_ecf'], abs=TOLERANCE)


def test_image_to_ground_hae(sicd, caplog):
    # project scp pixel (HAE)
    scp_ecef1 = point_projection.image_to_ground(sicd['scp_pixel'],
                                                 sicd['structure'],
                                                 projection_type='HAE')
    assert scp_ecef1 == pytest.approx(sicd['scp_ecf'], abs=TOLERANCE)

    scp_ecef2 = point_projection.image_to_ground([sicd['scp_pixel'], sicd['scp_pixel']],
                                                 sicd['structure'],
                                                 block_size=1,
                                                 projection_type='HAE')
    assert np.all(np.abs(scp_ecef2[0] - scp_ecef1) < TOLERANCE)
    assert np.all(np.abs(scp_ecef2[1] - scp_ecef1) < TOLERANCE)

    # error max_iterations < 1
    point_projection.image_to_ground(sicd['scp_pixel'],
                                     sicd['structure'],
                                     projection_type='HAE',
                                     max_iterations=0)
    assert 'max_iterations must be a positive integer' in caplog.text
    # error max_iterations > 100
    point_projection.image_to_ground(sicd['scp_pixel'],
                                     sicd['structure'],
                                     projection_type='HAE',
                                     max_iterations=101)
    assert 'maximum allowed max_iterations is 100' in caplog.text


def test_image_to_ground_errors(sicd):
    # invalid im_points (empty)
    with pytest.raises(ValueError, match="final dimension of im_points must have length 2"):
        point_projection.image_to_ground([], sicd['structure'], projection_type='PLANE')

    # invalid im_points (None)
    with pytest.raises(ValueError, match="The argument cannot be None"):
        point_projection.image_to_ground(None, sicd['structure'], projection_type='PLANE')

    # invalid projection_type
    with pytest.raises(ValueError, match="Got unrecognized projection type INVALID_PLANE"):
        point_projection.image_to_ground(sicd['scp_pixel'], sicd['structure'], projection_type='INVALID_PLANE')

    # invalid gref
    with pytest.raises(ValueError, match="gref must have three elements"):
        point_projection.image_to_ground(sicd['scp_pixel'], sicd['structure'], gref=[0.0, 0.0], projection_type='PLANE')

    # invalid ugpn
    with pytest.raises(ValueError, match="ugpn must have three elements"):
        point_projection.image_to_ground(sicd['scp_pixel'], sicd['structure'], ugpn=[0.0, 0.0], projection_type='PLANE')

    # dem_interpolator is none
    with pytest.raises(ValueError, match="dem_interpolator is None"):
        point_projection.image_to_ground(sicd['scp_pixel'], sicd['structure'], projection_type='DEM')

    # dem_interpolator is not DEMInterpolator type
    with pytest.raises(TypeError, match="dem_interpolator is of unsupported type"):
        point_projection.image_to_ground(sicd['scp_pixel'], sicd['structure'], projection_type='DEM',
                                         dem_interpolator=sicd['scp_pixel'])


def test_image_to_ground_geo(sicd):
    # project scp pixel to geodetic
    scp_geo = point_projection.image_to_ground_geo(sicd['scp_pixel'], sicd['structure'])
    assert scp_geo == pytest.approx(sicd['scp_llh'], abs=TOLERANCE)


def test_image_to_ground_dem(sicd):
    interp = DEMInterpolator()
    with pytest.raises(NotImplementedError):
        point_projection.image_to_ground(sicd['scp_pixel'],
                                         sicd['structure'],
                                         projection_type='DEM',
                                         dem_interpolator=interp)


def test_ground_to_image(sicd):
    # project scp ecef to pixel
    scp_pixel1 = point_projection.ground_to_image(sicd['scp_ecf'], sicd['structure'])
    assert scp_pixel1[0] == pytest.approx(sicd['scp_pixel'], abs=TOLERANCE)
    assert scp_pixel1[1] == pytest.approx(0.0, abs=TOLERANCE)

    scp_pixel2 = point_projection.ground_to_image([sicd['scp_ecf'], sicd['scp_ecf']], sicd['structure'], block_size=1)
    assert np.all(np.abs(scp_pixel2[0][0] - scp_pixel1[0]) < TOLERANCE)
    assert np.all(np.abs(scp_pixel2[0][1] - scp_pixel1[0]) < TOLERANCE)


def test_ground_to_image_geo(sicd):
    # project scp pixel to geodetic
    scp_pixel1 = point_projection.ground_to_image_geo(sicd['scp_llh'], sicd['structure'])
    assert scp_pixel1[0] == pytest.approx(sicd['scp_pixel'], abs=TOLERANCE)
    assert scp_pixel1[1] == pytest.approx(0.0, abs=TOLERANCE)


def test_image_to_ground_sidd(sidd, caplog):
    # project SIDD reference point pixel
    ref_point = [sidd.Measurement.PlaneProjection.ReferencePoint.Point.Row,
                 sidd.Measurement.PlaneProjection.ReferencePoint.Point.Col]
    scp_ecef1 = point_projection.image_to_ground(ref_point, sidd)

    sidd_ecef = sidd.Measurement.PlaneProjection.ReferencePoint.ECEF
    assert sidd_ecef.X == pytest.approx(scp_ecef1[0], abs=TOLERANCE)
    assert sidd_ecef.Y == pytest.approx(scp_ecef1[1], abs=TOLERANCE)
    assert sidd_ecef.Z == pytest.approx(scp_ecef1[2], abs=TOLERANCE)

    # force path through _get_outward_norm
    scp_ecef2 = point_projection.image_to_ground(ref_point, sidd, projection_type='PLANE', ugpn=None)
    assert scp_ecef2[0] == pytest.approx(sidd_ecef.X, abs=TOLERANCE)
    assert scp_ecef2[1] == pytest.approx(sidd_ecef.Y, abs=TOLERANCE)
    assert scp_ecef2[2] == pytest.approx(sidd_ecef.Z, abs=TOLERANCE)

    point_projection.image_to_ground(ref_point, sidd, tolerance=1e-13)
    assert 'minimum allowed tolerance is 1e-12' in caplog.text


def test_ground_to_image_sidd(sidd, caplog):
    # project reference point ecef to pixel
    ref_point = [sidd.Measurement.PlaneProjection.ReferencePoint.ECEF.X,
                 sidd.Measurement.PlaneProjection.ReferencePoint.ECEF.Y,
                 sidd.Measurement.PlaneProjection.ReferencePoint.ECEF.Z]
    scp_pixel1 = point_projection.ground_to_image(ref_point, sidd)

    sidd_rowcol = sidd.Measurement.PlaneProjection.ReferencePoint.Point
    assert sidd_rowcol.Row == pytest.approx(scp_pixel1[0][0], abs=TOLERANCE)
    assert sidd_rowcol.Col == pytest.approx(scp_pixel1[0][1], abs=TOLERANCE)
    assert scp_pixel1[1] == pytest.approx(0.0, abs=TOLERANCE)

    point_projection.ground_to_image(ref_point, sidd, tolerance=1e-13)
    assert 'minimum allowed tolerance is 1e-12' in caplog.text


def test_coa_projection(sicd, sidd):
    # smoke test
    proj = point_projection.COAProjection.from_sicd(sicd['structure'])
    assert proj.delta_arp is not None
    assert proj.delta_varp is not None
    assert proj.range_bias is not None
    assert proj.delta_range is not None

    # force path through RIC_ECF code (SICD)
    proj = point_projection.COAProjection.from_sicd(sicd['structure'], adj_params_frame='RIC_ECF')
    assert proj.delta_arp is not None

    # force path through RIC_ECF code (SIDD)
    proj = point_projection.COAProjection.from_sidd(sidd, adj_params_frame='RIC_ECF')
    assert proj.delta_arp is not None

    method_projection = point_projection._get_sicd_type_specific_projection(sicd['structure'])
    with pytest.raises(TypeError, match="time_coa_poly must be a Poly2DType instance"):
        point_projection.COAProjection(sicd['structure'].Position.ARPPoly,
                                       sicd['structure'].Position.ARPPoly,
                                       method_projection)
    with pytest.raises(TypeError, match="arp_poly must be an XYZPolyType instance"):
        point_projection.COAProjection(sicd['structure'].Grid.TimeCOAPoly,
                                       sicd['structure'].Grid.TimeCOAPoly,
                                       method_projection)
    with pytest.raises(TypeError, match="method_projection must be callable"):
        point_projection.COAProjection(sicd['structure'].Grid.TimeCOAPoly,
                                       sicd['structure'].Position.ARPPoly,
                                       'WRONG')


def test_validate_coords_error(sicd):
    # ECF coordinates must have length 3
    coords = np.array([[sicd['structure'].GeoData.SCP.ECF.X],
                      [sicd['structure'].GeoData.SCP.ECF.Y],
                      [sicd['structure'].GeoData.SCP.ECF.Z],
                      [sicd['structure'].GeoData.SCP.ECF.Z]])
    with pytest.raises(ValueError, match="final dimension of coords must have length 3"):
        point_projection._validate_coords(coords)


def test_validate_adjustment_param_error(sicd):
    # ECF coordinates must have length 3 and passed as an array
    coords = [[sicd['structure'].GeoData.SCP.ECF.X],
              [sicd['structure'].GeoData.SCP.ECF.Y],
              [sicd['structure'].GeoData.SCP.ECF.Z],
              [sicd['structure'].GeoData.SCP.ECF.Z]]
    with pytest.raises(ValueError, match=r"position must have shape \(3, \). Got \(4, 1\)"):
        point_projection._validate_adj_param(coords, 'position')

    coords = np.array(coords)
    with pytest.raises(ValueError, match=r"position must have shape \(3, \). Got \(4, 1\)"):
        point_projection._validate_adj_param(coords, 'position')


def test_image_to_ground_adjustable_offsets(sicd):
    # RIC_ECF and ECF frames give consistent results
    delta_arp_radial = 1000
    scp_perturb_ric = point_projection.image_to_ground(sicd['scp_pixel'], sicd['structure'],
                                                       projection_type='PLANE',
                                                       delta_arp=[delta_arp_radial, 0, 0],
                                                       delta_varp=[0, 0, 0],
                                                       adj_params_frame='RIC_ECF')

    arppos = sicd['structure'].SCPCOA.ARPPos.get_array()
    delta_arp_ecf = delta_arp_radial * (arppos / np.linalg.norm(arppos))
    scp_perturb_ecf = point_projection.image_to_ground(sicd['scp_pixel'], sicd['structure'],
                                                       projection_type='PLANE',
                                                       delta_arp=delta_arp_ecf,
                                                       delta_varp=[0, 0, 0],
                                                       adj_params_frame='ECF')
    assert np.allclose(scp_perturb_ric, scp_perturb_ecf)
    assert not np.allclose(sicd['scp_ecf'], scp_perturb_ecf)


def test_image_to_slant_sensitivity(sicd):
    assert sicd['structure'].Grid.ImagePlane == 'SLANT'
    m_spxy_il = point_projection.image_to_slant_sensitivity(sicd['structure'],
                                                            min(1.0, sicd['structure'].Grid.Row.SS),
                                                            min(1.0, sicd['structure'].Grid.Col.SS))
    # sensitivity when image plane is already slant should be nearly -identity due to relative orientation of slant and
    # image plane vectors
    assert np.allclose(m_spxy_il, -np.eye(2), atol=1e-3)
