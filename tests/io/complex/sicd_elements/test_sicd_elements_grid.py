#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np

from sarpy.io.complex.sicd_elements import Grid
from sarpy.io.xml.base import parse_xml_from_string


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

    # Define the weight function, so we can get the slant plane area
    new_row.define_weight_function(weight_size=512, populate=True)
    assert np.all(new_row.WgtFunct == 1.0)

    grid.Row = new_row
    grid.Col = new_row
    area = grid.get_slant_plane_area()
    assert isinstance(area, float)


def test_grid_wgttype(kwargs):
    """Test Grid.WgtTypeType class"""
    name = 'UNIFORM'
    params = {'fake_params': 'this_fake_str',
              'fake_params1': 'another_fake_str'}

    # Check basic WgtTypeType instantiation
    weight = Grid.WgtTypeType(WindowName=name, Parameters=params, **kwargs)
    assert weight._xml_ns == kwargs['_xml_ns']
    assert weight._xml_ns_key == kwargs['_xml_ns_key']

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
