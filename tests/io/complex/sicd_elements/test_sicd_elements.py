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
from sarpy.io.complex.sicd_elements import ImageData
from sarpy.io.complex.sicd_elements import PFA
from sarpy.io.complex.sicd_elements import RadarCollection
from sarpy.io.complex.sicd_elements import Radiometric
from sarpy.io.complex.sicd_elements import RgAzComp
from sarpy.io.complex.sicd_elements import Timeline
from sarpy.io.complex.sicd_elements import utils
from sarpy.io.complex.sicd_elements import validation_checks
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.xml.base import parse_xml_from_string
TOLERANCE = 1e-8
KWARGS = {'_xml_ns': 'ns', '_xml_ns_key': 'key'}


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

    made_up_poly = blocks.Poly2DType([[10, 5], [8, 3], [1, 0.5]])
    stdeskew = PFA.STDeskewType(False, made_up_poly, **KWARGS)
    assert stdeskew.Applied is False
    assert stdeskew.STDSPhasePoly == made_up_poly
    assert stdeskew._xml_ns == KWARGS['_xml_ns']
    assert stdeskew._xml_ns_key == KWARGS['_xml_ns_key']

    # Nominal instantiation
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
                          **KWARGS)
    assert isinstance(pfa_nom, PFA.PFAType)
    assert pfa_nom._xml_ns == KWARGS['_xml_ns']
    assert pfa_nom._xml_ns_key == KWARGS['_xml_ns_key']
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
                             **KWARGS)
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

    # Check basic WgtTypeType instantiation
    weight = Grid.WgtTypeType(WindowName=name, Parameters=params, **KWARGS)
    assert weight._xml_ns == KWARGS['_xml_ns']
    assert weight._xml_ns_key == KWARGS['_xml_ns_key']

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


def test_radiometric(sicd, rma_sicd):
    noise_level = Radiometric.NoiseLevelType_()
    assert noise_level.NoiseLevelType is None
    assert noise_level.NoisePoly is None

    noise_level = Radiometric.NoiseLevelType_('ABSOLUTE')
    assert noise_level.NoiseLevelType == 'ABSOLUTE'
    assert noise_level.NoisePoly is None

    noise_level = Radiometric.NoiseLevelType_(None, rma_sicd.Radiometric.NoiseLevel.NoisePoly, **KWARGS)
    assert noise_level.NoiseLevelType == 'RELATIVE'

    noise_level = Radiometric.NoiseLevelType_(None, sicd.Radiometric.NoiseLevel.NoisePoly, **KWARGS)
    assert noise_level._xml_ns == KWARGS['_xml_ns']
    assert noise_level._xml_ns_key == KWARGS['_xml_ns_key']
    assert noise_level.NoiseLevelType == 'ABSOLUTE'
    assert noise_level.NoisePoly == sicd.Radiometric.NoiseLevel.NoisePoly

    radio_type = Radiometric.RadiometricType()
    assert radio_type.NoiseLevel is None
    assert radio_type.RCSSFPoly is None
    assert radio_type.SigmaZeroSFPoly is None
    assert radio_type.BetaZeroSFPoly is None
    assert radio_type.GammaZeroSFPoly is None

    radio_type = Radiometric.RadiometricType(noise_level, sicd.Radiometric.RCSSFPoly)
    assert radio_type.NoiseLevel == noise_level
    assert radio_type.RCSSFPoly == sicd.Radiometric.RCSSFPoly

    radio_type._derive_parameters(sicd.Grid, sicd.SCPCOA)
    assert radio_type.SigmaZeroSFPoly is not None
    assert radio_type.BetaZeroSFPoly is not None
    assert radio_type.GammaZeroSFPoly is not None

    radio_type = Radiometric.RadiometricType(noise_level)
    assert radio_type.NoiseLevel == noise_level
    assert radio_type.RCSSFPoly is None
    assert radio_type.SigmaZeroSFPoly is None
    assert radio_type.BetaZeroSFPoly is None
    assert radio_type.GammaZeroSFPoly is None

    radio_type._derive_parameters(sicd.Grid, sicd.SCPCOA)
    assert radio_type.NoiseLevel == noise_level
    assert radio_type.RCSSFPoly is None
    assert radio_type.SigmaZeroSFPoly is None
    assert radio_type.BetaZeroSFPoly is None
    assert radio_type.GammaZeroSFPoly is None


def test_rgazcomp(rma_sicd):
    rg_az = RgAzComp.RgAzCompType(None, None, **KWARGS)
    rg_az._derive_parameters(rma_sicd.Grid, rma_sicd.Timeline, rma_sicd.SCPCOA)
    assert rg_az._xml_ns == KWARGS['_xml_ns']
    assert rg_az._xml_ns_key == KWARGS['_xml_ns_key']
    assert rg_az.AzSF is not None
    assert rg_az.KazPoly is not None


def test_timeline(sicd):
    def get_ipp_set(ipp, idx, **kwargs):
        return Timeline.IPPSetType(ipp.TStart,
                                   ipp.TEnd,
                                   ipp.IPPStart,
                                   ipp.IPPEnd,
                                   ipp.IPPPoly,
                                   idx,
                                   **kwargs)

    ippset1 = get_ipp_set(sicd.Timeline.IPP[0], 1, **KWARGS)
    assert ippset1.TStart == sicd.Timeline.IPP[0].TStart
    assert ippset1.TEnd == sicd.Timeline.IPP[0].TEnd
    assert ippset1.IPPStart == sicd.Timeline.IPP[0].IPPStart
    assert ippset1.IPPEnd == sicd.Timeline.IPP[0].IPPEnd
    assert ippset1.IPPPoly == sicd.Timeline.IPP[0].IPPPoly
    assert ippset1._basic_validity_check()

    ippset2 = get_ipp_set(sicd.Timeline.IPP[1], 2, **KWARGS)
    ipp_list = [ippset1, ippset2]

    timeline = Timeline.TimelineType(sicd.Timeline.CollectStart,
                                     sicd.Timeline.CollectDuration,
                                     ipp_list)
    assert timeline.CollectStart == sicd.Timeline.CollectStart
    assert timeline.CollectDuration == sicd.Timeline.CollectDuration

    for idx in np.arange(len(timeline.IPP)):
        assert timeline.IPP[idx].TStart == ipp_list[idx].TStart
        assert timeline.IPP[idx].TEnd == ipp_list[idx].TEnd
        assert timeline.IPP[idx].IPPStart == ipp_list[idx].IPPStart
        assert timeline.IPP[idx].IPPEnd == ipp_list[idx].IPPEnd
        assert timeline.IPP[idx].IPPPoly == ipp_list[idx].IPPPoly

    assert timeline.CollectEnd == timeline.CollectStart + np.timedelta64(int(timeline.CollectDuration*1e6), 'us')
    assert timeline._check_ipp_consecutive()
    assert timeline._check_ipp_times()
    assert timeline._basic_validity_check()

    # Check bail out for IPP consecutive with a single IPP set
    timeline = Timeline.TimelineType(sicd.Timeline.CollectStart,
                                     sicd.Timeline.CollectDuration,
                                     ipp_list[0])
    assert timeline._check_ipp_consecutive()


def test_timeline_start_end_mismatches(sicd, caplog):
    # Swap Tstart and TEnd along with IPPStart and IPPEnd
    ipp0 = sicd.Timeline.IPP[0]
    bad_ippset = Timeline.IPPSetType(ipp0.TEnd,
                                     ipp0.TStart,
                                     ipp0.IPPEnd,
                                     ipp0.IPPStart,
                                     ipp0.IPPPoly,
                                     1,
                                     **KWARGS)
    bad_ippset._basic_validity_check()
    assert 'TStart ({}) >= TEnd ({})'.format(ipp0.TEnd, ipp0.TStart) in caplog.text
    assert 'IPPStart ({}) >= IPPEnd ({})'.format(ipp0.IPPEnd, ipp0.IPPStart) in caplog.text

    # CollectEnd is None with no CollectStart
    bad_timeline = Timeline.TimelineType(None, sicd.Timeline.CollectDuration, bad_ippset)
    assert bad_timeline.CollectEnd is None


def test_timeline_negative_tstart(sicd, caplog):
    # Negative TStart
    ipp0 = sicd.Timeline.IPP[0]
    bad_ippset = Timeline.IPPSetType(-ipp0.TStart,
                                     ipp0.TEnd,
                                     ipp0.IPPStart,
                                     ipp0.IPPEnd,
                                     ipp0.IPPPoly,
                                     1,
                                     **KWARGS)
    bad_timeline = Timeline.TimelineType(None, sicd.Timeline.CollectDuration, bad_ippset)
    bad_timeline._check_ipp_times()
    assert 'IPP entry 0 has negative TStart' in caplog.text


def test_timeline_bad_tend(sicd, caplog):
    # TEnd too large
    ipp0 = sicd.Timeline.IPP[0]
    bad_ippset = Timeline.IPPSetType(ipp0.TStart,
                                     sicd.Timeline.CollectDuration+1,
                                     ipp0.IPPStart,
                                     ipp0.IPPEnd,
                                     ipp0.IPPPoly,
                                     1,
                                     **KWARGS)
    bad_timeline = Timeline.TimelineType(None, sicd.Timeline.CollectDuration, bad_ippset)
    bad_timeline._check_ipp_times()
    assert 'appreciably larger than CollectDuration' in caplog.text


def test_timeline_unset_ipp(sicd):
    # Check times is True with no IPP set
    bad_timeline = Timeline.TimelineType(sicd.Timeline.CollectStart, sicd.Timeline.CollectDuration, None)
    assert bad_timeline._check_ipp_times()


def test_imagedata(sicd, caplog):
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

    image_type1 = ImageData.FullImageType(sicd.ImageData.NumRows, sicd.ImageData.NumCols, **KWARGS)
    assert image_type1._xml_ns == KWARGS['_xml_ns']
    assert image_type1._xml_ns_key == KWARGS['_xml_ns_key']
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
                                         **KWARGS)
    assert image_data._xml_ns == KWARGS['_xml_ns']
    assert image_data._xml_ns_key == KWARGS['_xml_ns_key']
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
                                         **KWARGS)
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
                                         **KWARGS)
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


def test_radarcollection_getbandname():
    assert RadarCollection.get_band_name(None) == 'UN'
    assert RadarCollection.get_band_name(3.5e6) == 'HF'
    assert RadarCollection.get_band_name(3.5e7) == 'VHF'
    assert RadarCollection.get_band_name(3.5e8) == 'UHF'
    assert RadarCollection.get_band_name(1.5e9) == 'L'
    assert RadarCollection.get_band_name(2.5e9) == 'S'
    assert RadarCollection.get_band_name(4.5e9) == 'C'
    assert RadarCollection.get_band_name(8.5e9) == 'X'
    assert RadarCollection.get_band_name(1.5e10) == 'KU'
    assert RadarCollection.get_band_name(2.5e10) == 'K'
    assert RadarCollection.get_band_name(3.0e10) == 'KA'
    assert RadarCollection.get_band_name(6.0e10) == 'V'
    assert RadarCollection.get_band_name(1.0e11) == 'W'
    assert RadarCollection.get_band_name(2.0e11) == 'MM'
    assert RadarCollection.get_band_name(5.0e11) == 'UN'


def test_radarcollection_txfreqtype(sicd, caplog):

    bad_tx_freq = RadarCollection.TxFrequencyType(None,
                                                  sicd.RadarCollection.TxFrequency.Max,
                                                  **KWARGS)
    assert bad_tx_freq.center_frequency is None

    bad_tx_freq = RadarCollection.TxFrequencyType(sicd.RadarCollection.TxFrequency.Max,
                                                  sicd.RadarCollection.TxFrequency.Min,
                                                  **KWARGS)
    assert not bad_tx_freq._basic_validity_check()
    assert 'Invalid frequency bounds Min ({}) > Max ({})'.format(bad_tx_freq.Min, bad_tx_freq.Max) in caplog.text

    tx_freq = RadarCollection.TxFrequencyType(sicd.RadarCollection.TxFrequency.Min,
                                              sicd.RadarCollection.TxFrequency.Max,
                                              **KWARGS)
    assert tx_freq._xml_ns == KWARGS['_xml_ns']
    assert tx_freq._xml_ns_key == KWARGS['_xml_ns_key']
    assert tx_freq.Min == sicd.RadarCollection.TxFrequency.Min
    assert tx_freq.Max == sicd.RadarCollection.TxFrequency.Max
    assert tx_freq.center_frequency == 0.5 * (tx_freq.Min + tx_freq.Max)

    tx_freq._apply_reference_frequency(1000)
    assert tx_freq.Min == sicd.RadarCollection.TxFrequency.Min + 1000
    assert tx_freq.Max == sicd.RadarCollection.TxFrequency.Max + 1000
    assert tx_freq._basic_validity_check()
    assert tx_freq.get_band_abbreviation() == 'X__'

    tx_freq_arr = tx_freq.get_array()
    assert np.all(tx_freq_arr == np.array([tx_freq.Min, tx_freq.Max]))

    assert tx_freq.from_array(None) is None
    tx_freq1 = tx_freq.from_array(tx_freq_arr)
    assert tx_freq1.Min == tx_freq.Min
    assert tx_freq1.Max == tx_freq.Max

    with pytest.raises(ValueError, match='Expected array to be of length 2, and received 1'):
        tx_freq.from_array([tx_freq_arr[0]])
    with pytest.raises(ValueError, match='Expected array to be numpy.ndarray, list, or tuple'):
        tx_freq.from_array(tx_freq)


def test_radarcollection_waveformparamtype():
    rc = RadarCollection.WaveformParametersType(RcvDemodType='STRETCH', RcvFMRate=1.0)
    assert rc.RcvFMRate == 1.0

    rc = RadarCollection.WaveformParametersType(RcvDemodType='CHIRP', RcvFMRate=None)
    assert rc.RcvFMRate == 0.0

    rc = RadarCollection.WaveformParametersType(RcvDemodType='cHiRp', RcvFMRate=None)
    assert rc.RcvFMRate is None
