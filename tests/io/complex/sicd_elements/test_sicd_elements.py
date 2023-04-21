#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import copy
import pathlib

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import blocks
from sarpy.io.complex.sicd_elements import Grid
from sarpy.io.complex.sicd_elements import PFA
from sarpy.io.complex.sicd_elements import utils
from sarpy.io.complex.sicd_elements import validation_checks
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.xml.base import parse_xml_from_string
TOLERANCE = 1e-8


@pytest.fixture(scope='module')
def sicd():
    xml_file = pathlib.Path(pathlib.Path.cwd(), 'tests/data/example.sicd.xml')
    structure = SICDType().from_xml_file(xml_file)

    return structure


@pytest.fixture(scope='module')
def rma_sicd():
    xml_file = pathlib.Path(pathlib.Path.cwd(), 'tests/data/example.sicd.rma.xml')
    structure = SICDType().from_xml_file(xml_file)

    return structure


def test_validation_checks(sicd):
    """Smoke test with PFA SICD"""
    validation_checks.detailed_validation_checks(sicd)


def test_validation_checks_with_rma(rma_sicd):
    """Smoke test with RMA SICD"""
    validation_checks.detailed_validation_checks(rma_sicd)


def test_utils(sicd, rma_sicd, caplog):
    """Check sicd_elements utility functions"""
    assert utils.is_same_sensor(sicd, rma_sicd)
    assert utils.is_same_sensor(sicd, sicd)

    assert utils.is_same_start_time(sicd, rma_sicd)
    assert utils.is_same_start_time(sicd, sicd)

    assert not utils.is_same_size(sicd, rma_sicd)
    assert utils.is_same_size(sicd, sicd)

    assert utils.is_same_duration(sicd, rma_sicd)
    assert utils.is_same_duration(sicd, sicd)

    assert utils.is_same_scp(sicd, rma_sicd)
    assert utils.is_same_scp(sicd, sicd)

    assert not utils.is_same_band(sicd, rma_sicd)
    assert utils.is_same_band(sicd, sicd)

    assert not utils.is_general_match(sicd, rma_sicd)
    assert utils.is_general_match(sicd, sicd)

    pol = utils.polstring_version_required(None)
    assert pol == (1, 1, 0)
    pol = utils.polstring_version_required('V:V:H')
    assert 'Expected polarization string of length 2, but populated as `3`' in caplog.text
    assert pol is None
    pol = utils.polstring_version_required('V')
    assert 'Expected polarization string of length 2, but populated as `1`' in caplog.text
    assert pol is None
    pol = utils.polstring_version_required('S:V')
    assert pol == (1, 3, 0)
    pol = utils.polstring_version_required('H:X')
    assert pol == (1, 3, 0)
    pol = utils.polstring_version_required('V:RHC')
    assert pol == (1, 2, 1)
    pol = utils.polstring_version_required('LHC:H')
    assert pol == (1, 2, 1)
    pol = utils.polstring_version_required('V:H')
    assert pol == (1, 1, 0)
    pol = utils.polstring_version_required('OTHER:H')
    assert pol == (1, 3, 0)
    pol = utils.polstring_version_required('H:OTHERpol')
    assert pol == (1, 3, 0)

    # Must have both ImageFormation and RadarCollection
    freq = utils._get_center_frequency(None, sicd.ImageFormation)
    assert freq is None
    freq = utils._get_center_frequency(sicd.RadarCollection, None)
    assert freq is None

    # Use a copy to change RefFreqIndex value
    radar_collection = copy.copy(sicd.RadarCollection)
    radar_collection.RefFreqIndex = 10
    freq = utils._get_center_frequency(radar_collection, sicd.ImageFormation)
    assert freq is None

    radar_collection.RefFreqIndex = None
    freq = utils._get_center_frequency(radar_collection, sicd.ImageFormation)
    assert freq == sicd.ImageFormation.TxFrequencyProc.center_frequency


def test_pfa(sicd):
    """Test PFA classes"""
    stdeskew = PFA.STDeskewType()
    assert isinstance(stdeskew, PFA.STDeskewType)

    kwargs = {'_xml_ns': 'ns', '_xml_ns_key': 'key'}
    made_up_poly = blocks.Poly2DType([[10, 5], [8, 3], [1, 0.5]])
    stdeskew = PFA.STDeskewType(False, made_up_poly, **kwargs)
    assert stdeskew.Applied is False
    assert stdeskew.STDSPhasePoly == made_up_poly
    assert stdeskew._xml_ns == 'ns'
    assert stdeskew._xml_ns_key == 'key'

    #Nominal instantiation
    pfa_nom = PFA.PFAType(sicd.PFA.FPN,
                          sicd.PFA.IPN,
                          sicd.PFA.PolarAngRefTime,
                          sicd.PFA.PolarAngPoly,
                          sicd.PFA.SpatialFreqSFPoly,
                          sicd.PFA.Krg1,
                          sicd.PFA.Krg2,
                          sicd.PFA.Kaz1,
                          sicd.PFA.Kaz2,
                          stdeskew,
                          **kwargs)
    assert isinstance(pfa_nom, PFA.PFAType)
    assert pfa_nom._xml_ns == 'ns'
    assert pfa_nom._xml_ns_key == 'key'
    assert pfa_nom._basic_validity_check()
    assert pfa_nom._check_polar_ang_ref()

    # No PolarAnglePoly path
    pfa_no_pap = PFA.PFAType(sicd.PFA.FPN,
                             sicd.PFA.IPN,
                             sicd.PFA.PolarAngRefTime,
                             None,
                             sicd.PFA.SpatialFreqSFPoly,
                             sicd.PFA.Krg1,
                             sicd.PFA.Krg2,
                             sicd.PFA.Kaz1,
                             sicd.PFA.Kaz2,
                             stdeskew,
                             **kwargs)
    assert pfa_no_pap._check_polar_ang_ref()

    # Populate empty PFAType with sicd components after instantiation
    pfa_empty = PFA.PFAType()
    pfa_empty._derive_parameters(sicd.Grid, sicd.SCPCOA, sicd.GeoData, sicd.Position, sicd.Timeline)
    assert pfa_empty.PolarAngRefTime == pytest.approx(sicd.SCPCOA.SCPTime, abs=TOLERANCE)
    assert isinstance(pfa_empty.IPN, blocks.XYZType)
    assert isinstance(pfa_empty.FPN, blocks.XYZType)
    assert isinstance(pfa_empty.PolarAngPoly, blocks.Poly1DType)
    assert isinstance(pfa_empty.SpatialFreqSFPoly, blocks.Poly1DType)
    assert isinstance(pfa_empty.Krg1, float)
    assert isinstance(pfa_empty.Krg2, float)
    assert isinstance(pfa_empty.Kaz1, float)
    assert isinstance(pfa_empty.Kaz2, float)

    # Try it without GeoData
    pfa_empty_no_geo = PFA.PFAType()
    pfa_empty_no_geo._derive_parameters(sicd.Grid, sicd.SCPCOA, None, sicd.Position, sicd.Timeline)
    assert pfa_empty_no_geo.PolarAngRefTime == pytest.approx(sicd.SCPCOA.SCPTime, abs=TOLERANCE)
    assert pfa_empty_no_geo.IPN is None
    assert pfa_empty_no_geo.FPN is None
    assert pfa_empty_no_geo.PolarAngPoly is None
    assert pfa_empty_no_geo.SpatialFreqSFPoly is None
    assert pfa_empty_no_geo.Krg1 is None
    assert pfa_empty_no_geo.Krg2 is None
    assert pfa_empty_no_geo.Kaz1 is None
    assert pfa_empty_no_geo.Kaz2 is None

    # Without FPN to test that path
    pfa_no_fpn = PFA.PFAType(None,
                             sicd.PFA.IPN,
                             sicd.PFA.PolarAngRefTime,
                             sicd.PFA.PolarAngPoly,
                             sicd.PFA.SpatialFreqSFPoly,
                             sicd.PFA.Krg1,
                             sicd.PFA.Krg2,
                             sicd.PFA.Kaz1,
                             sicd.PFA.Kaz2,
                             stdeskew)
    assert pfa_no_fpn.pfa_polar_coords(sicd.Position,
                                       sicd.GeoData.SCP.ECF[:],
                                       0.0) == (None, None)
    assert pfa_no_fpn.pfa_polar_coords(sicd.Position,
                                       sicd.GeoData.SCP.ECF[:],
                                       np.array([6378137.0, 0])) == (None, None)


def test_grid_gridtype(sicd, rma_sicd, caplog):
    """Test Grid.GridType class"""
    # Start with empty Grid and populate values
    grid_empty = Grid.GridType()
    grid_empty._derive_time_coa_poly(sicd.CollectionInfo, sicd.SCPCOA)
    assert grid_empty.TimeCOAPoly.Coefs[0][0] == sicd.SCPCOA.SCPTime

    grid_empty._derive_rg_az_comp(sicd.GeoData,
                                  sicd.SCPCOA,
                                  sicd.RadarCollection,
                                  sicd.ImageFormation)
    assert grid_empty.ImagePlane == 'SLANT'
    assert grid_empty.Type == 'RGAZIM'
    assert grid_empty.Row is not None
    assert grid_empty.Col is not None

    # Incorrect ImagePlane option
    grid_empty.ImagePlane = 'SLAT'
    grid_empty._derive_rg_az_comp(sicd.GeoData,
                                  sicd.SCPCOA,
                                  sicd.RadarCollection,
                                  sicd.ImageFormation)
    assert 'Image Formation Algorithm is RgAzComp, which requires "SLANT"' in caplog.text
    assert grid_empty.ImagePlane == 'SLANT'

    # Incorrect Type option
    grid_empty.Type = 'RGAZ'
    grid_empty._derive_rg_az_comp(sicd.GeoData,
                                  sicd.SCPCOA,
                                  sicd.RadarCollection,
                                  sicd.ImageFormation)
    assert 'Image Formation Algorithm is RgAzComp, which requires "RGAZIM"' in caplog.text
    assert grid_empty.Type == 'RGAZIM'

    # Force KCtr=None path through _derive_rg_az_comp
    grid_empty.Row.KCtr = None
    grid_empty.Col.KCtr = None
    grid_empty._derive_rg_az_comp(sicd.GeoData,
                                  sicd.SCPCOA,
                                  sicd.RadarCollection,
                                  sicd.ImageFormation)
    assert grid_empty.Row.KCtr is not None
    assert grid_empty.Col.KCtr is not None

    # Force KCtr=None path through _derive_pfa
    grid_empty.Row.KCtr = None
    grid_empty.Col.KCtr = None
    grid_empty._derive_pfa(sicd.GeoData,
                           sicd.RadarCollection,
                           sicd.ImageFormation,
                           sicd.Position,
                           sicd.PFA)
    assert grid_empty.Row.KCtr is not None
    assert grid_empty.Col.KCtr is not None

    # Force unit vector derivation path through _derive_pfa
    grid_empty.Row.UVectECF = None
    grid_empty.Col.UVectECF = None
    grid_empty._derive_pfa(sicd.GeoData,
                           sicd.RadarCollection,
                           sicd.ImageFormation,
                           sicd.Position,
                           sicd.PFA)
    assert grid_empty.Row.UVectECF is not None
    assert grid_empty.Col.UVectECF is not None

    # Force unit vector derivation path through _derive_rma
    grid_empty.Row.UVectECF = None
    grid_empty.Col.UVectECF = None
    grid_empty._derive_rma(rma_sicd.RMA,
                           rma_sicd.GeoData,
                           rma_sicd.RadarCollection,
                           rma_sicd.ImageFormation,
                           rma_sicd.Position)
    assert grid_empty.Row.UVectECF is not None
    assert grid_empty.Col.UVectECF is not None

    grid = Grid.GridType(sicd.Grid.ImagePlane,
                         sicd.Grid.Type,
                         sicd.Grid.TimeCOAPoly,
                         sicd.Grid.Row,
                         sicd.Grid.Col)
    grid.derive_direction_params(sicd.ImageData, populate=True)

    # Basic validity
    assert grid._basic_validity_check()

    # Resolution abbreviation checks
    expected_abbr = int(100 * (abs(sicd.Grid.Row.ImpRespWid) *
                               abs(sicd.Grid.Col.ImpRespWid))**0.5)
    assert grid.get_resolution_abbreviation() == '{0:04d}'.format(expected_abbr)

    grid.Row.ImpRespWid = None
    grid.Col.ImpRespWid = None
    assert grid.get_resolution_abbreviation() == '0000'

    grid.Row.ImpRespWid = 100
    grid.Col.ImpRespWid = 100
    assert grid.get_resolution_abbreviation() == '9999'

    # Create a new row type with minimum input and WgtType defined
    new_row = Grid.DirParamType(ImpRespBW=0.88798408351600244,
                                WgtType=Grid.WgtTypeType(WindowName='UNIFORM'))

    # Define the weight function so we can get the slant plane area
    new_row.define_weight_function(weight_size=512, populate=True)
    assert np.all(new_row.WgtFunct == 1.0)

    grid.Row = new_row
    grid.Col = new_row
    area = grid.get_slant_plane_area()
    assert isinstance(area, float)


def test_grid_wgttype():
    """Test Grid.WgtTypeType class"""
    name = 'UNIFORM'
    params = {'fake_params': 'this_fake_str',
              'fake_params1': 'another_fake_str'}
    kwargs = {'_xml_ns': 'ns', '_xml_ns_key': 'key'}

    # Check basic WgtTypeType instantiation
    weight = Grid.WgtTypeType(WindowName=name, Parameters=params, **kwargs)
    assert weight._xml_ns == 'ns'
    assert weight._xml_ns_key == 'key'

    # get_parameter_value checks
    assert weight.get_parameter_value('fake_params1') == params['fake_params1']

    # Passing None for value returns first parameter
    assert weight.get_parameter_value(None) == params['fake_params']

    # No parameters
    weight1 = Grid.WgtTypeType(WindowName=name)
    assert weight1.get_parameter_value('fake_params') is None

    node_str = '''
        <WgtType>
            <WindowName>Taylor</WindowName>
            <Parameter name="nbar">5</Parameter>
            <Parameter name="sll_db">-35.0</Parameter>
            <Parameter name="osf">1.2763784887678868</Parameter>
        </WgtType>
    '''
    weight = Grid.WgtTypeType()
    node, ns = parse_xml_from_string(node_str)

    weight1 = weight.from_node(node, ns)
    assert isinstance(weight1, Grid.WgtTypeType)
    assert weight1.WindowName == 'Taylor'
    assert weight1.Parameters.get('nbar') == '5'
    assert weight1.Parameters.get('sll_db') == '-35.0'
    assert weight1.Parameters.get('osf') == '1.2763784887678868'
