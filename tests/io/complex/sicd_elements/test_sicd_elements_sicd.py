#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import copy
import re

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import RgAzComp


def test_sicd_smoke_tests(sicd, rma_sicd, tol):
    assert sicd.is_valid()

    sicd_copy = sicd.copy()
    assert sicd_copy.is_valid()

    sicd_copy.NITF = None
    assert sicd_copy.NITF == {}

    sicd_copy.NITF = {'test': 'TEST'}
    assert sicd_copy.NITF == {'test': 'TEST'}

    assert sicd_copy.ImageFormType == 'PFA'

    scp_ecf = [6378138., 0., 0.]
    scp_llh = [.01, 0., 0.]
    sicd_copy.update_scp(scp_ecf)
    assert sicd_copy.GeoData.SCP.ECF.X == 6378138.
    sicd_copy.update_scp(scp_llh, 'LLH')
    assert sicd_copy.GeoData.SCP.LLH.Lat == 0.01

    # Check nothing to do path
    sicd_copy.define_geo_image_corners()
    sicd_copy.derive()
    assert sicd_copy.get_transmit_band_name() == 'X'
    assert sicd_copy.get_processed_polarization_abbreviation() == 'VV'
    assert sicd_copy.get_processed_polarization() == 'V:V'

    # Take the RMA path through derive
    rma_sicd.derive()
    rma_sicd.RadarCollection.RefFreqIndex = 1
    rma_sicd.apply_reference_frequency(10000)
    assert np.all(rma_sicd.RadarCollection.TxFrequency.get_array() ==
                  sicd.RadarCollection.TxFrequency.get_array()+10000)

    row_res, col_res = sicd_copy.get_ground_resolution()
    assert row_res is not None
    assert col_res is not None

    assert sicd_copy.can_project_coordinates()

    assert sicd_copy.coa_projection is None
    sicd_copy.define_coa_projection(override=True)
    assert sicd_copy.coa_projection is not None

    scp_pixel, _, _ = sicd_copy.project_ground_to_image([sicd_copy.GeoData.SCP.ECF.X,
                                                         sicd_copy.GeoData.SCP.ECF.Y,
                                                         sicd_copy.GeoData.SCP.ECF.Z])
    assert scp_pixel[0] == pytest.approx(sicd_copy.ImageData.SCPPixel.Row, abs=tol)
    assert scp_pixel[1] == pytest.approx(sicd_copy.ImageData.SCPPixel.Col, abs=tol)

    scp_pixel, _, _ = sicd_copy.project_ground_to_image_geo([sicd_copy.GeoData.SCP.LLH.Lat,
                                                             sicd_copy.GeoData.SCP.LLH.Lon,
                                                             sicd_copy.GeoData.SCP.LLH.HAE])
    assert scp_pixel[0] == pytest.approx(sicd_copy.ImageData.SCPPixel.Row, abs=tol)
    assert scp_pixel[1] == pytest.approx(sicd_copy.ImageData.SCPPixel.Col, abs=tol)

    scp_ecef = sicd_copy.project_image_to_ground([sicd_copy.ImageData.SCPPixel.Row, sicd_copy.ImageData.SCPPixel.Col],
                                                 projection_type='PLANE')
    assert scp_ecef == pytest.approx([sicd_copy.GeoData.SCP.ECF.X,
                                      sicd_copy.GeoData.SCP.ECF.Y,
                                      sicd_copy.GeoData.SCP.ECF.Z], abs=tol)

    scp_geo = sicd_copy.project_image_to_ground_geo([sicd_copy.ImageData.SCPPixel.Row, sicd_copy.ImageData.SCPPixel.Col])
    assert scp_geo == pytest.approx([sicd_copy.GeoData.SCP.LLH.Lat,
                                     sicd_copy.GeoData.SCP.LLH.Lon,
                                     sicd_copy.GeoData.SCP.LLH.HAE], abs=tol)

    sicd_copy.populate_rniirs()
    assert sicd_copy.CollectionInfo.Parameters['INFORMATION_DENSITY'] is not None
    assert sicd_copy.CollectionInfo.Parameters['PREDICTED_RNIIRS'] is not None

    name = sicd_copy.get_suggested_name()
    assert name == 'SyntheticCore_001_184124_SL0099L_00N000E_001X___SVV_0101_SPY'

    version = sicd_copy.version_required()
    assert version == (1, 1, 0)

    details = sicd_copy.get_des_details()
    assert details['DESSHSI'] == 'SICD Volume 1 Design & Implementation Description Document'
    assert details['DESSHSV'] == '1.3.0'
    assert details['DESSHSD'] == '2021-11-30T00:00:00Z'
    assert details['DESSHTN'] == 'urn:SICD:1.3.0'

    xml_bytes = sicd.to_xml_bytes()
    assert isinstance(xml_bytes, bytes)

    xml_string = sicd.to_xml_string()
    assert isinstance(xml_string, str)
    sicd1 = sicd.from_xml_string(xml_string)
    assert sicd1.is_valid()

    sicd2, out_row_bounds, out_col_bounds = sicd.create_subset_structure()
    assert sicd2.ImageData.FirstRow == sicd.ImageData.FirstRow
    assert sicd2.ImageData.NumRows == sicd.ImageData.NumRows
    assert sicd2.ImageData.FirstCol == sicd.ImageData.FirstCol
    assert sicd2.ImageData.NumCols == sicd.ImageData.NumCols
    assert out_row_bounds == (0, sicd.ImageData.NumRows)
    assert out_col_bounds == (0, sicd.ImageData.NumCols)

    min_max_vals = (10, 100)
    sicd2, out_row_bounds, out_col_bounds = sicd.create_subset_structure(row_bounds=min_max_vals,
                                                                         column_bounds=min_max_vals)
    assert sicd2.ImageData.FirstRow == sicd.ImageData.FirstRow + min_max_vals[0]
    assert sicd2.ImageData.NumRows == min_max_vals[1] - min_max_vals[0]
    assert sicd2.ImageData.FirstCol == sicd.ImageData.FirstCol + min_max_vals[0]
    assert sicd2.ImageData.NumCols == min_max_vals[1] - min_max_vals[0]
    assert out_row_bounds == min_max_vals
    assert out_col_bounds == min_max_vals


def test_nitf_setter_failures(sicd):
    a_list = ['test', 'TEST']
    with pytest.raises(TypeError, match=f'data must be dictionary instance. Received {type(a_list)}'):
        sicd.NITF = a_list


def test_update_scp_failures(sicd):
    bad_scp = [0, 0, 0, 0]
    with pytest.raises(TypeError, match='point must be an numpy.ndarray'):
        sicd.update_scp({})
    with pytest.raises(ValueError, match='point must be a one-dimensional, 3 element array'):
        sicd.update_scp(bad_scp)


def test_smoke_test_derive_with_rgazcomp(sicd):
    sicd.ImageFormation.ImageFormAlgo = 'RGAZCOMP'
    sicd.derive()


def test_define_geo_image_corners(sicd):
    sicd.GeoData = None
    sicd.define_geo_image_corners()
    assert sicd.GeoData is not None


def test_define_geo_valid_data(sicd):
    sicd.GeoData.ValidData = None
    sicd.define_geo_valid_data()
    assert sicd.GeoData.ValidData is not None

    sicd.GeoData = None
    sicd.define_geo_valid_data()
    # Nothing to be done path
    assert sicd.GeoData is None


def test_missing_image_formation(sicd):
    sicd.ImageFormation = None
    assert sicd.get_transmit_band_name() == 'UN'
    assert sicd.get_processed_polarization_abbreviation() == 'UN'
    assert sicd.get_processed_polarization() == 'UN'


def test_apply_reference_frequency_errors(sicd):
    with pytest.raises(ValueError, match='RadarCollection.RefFreqIndex is not defined. '
                                         'The reference frequency should not be applied.'):
        sicd.apply_reference_frequency(10000)
    sicd.RadarCollection.RefFreqIndex = 1
    sicd.apply_reference_frequency(10000)

    sicd.RadarCollection = None
    with pytest.raises(ValueError, match='RadarCollection is not defined. The reference frequency cannot be applied.'):
        sicd.apply_reference_frequency(10000)


def test_create_subset_structure_errors(sicd):
    min_max_vals = (10, 100000)
    with pytest.raises(ValueError, match=re.escape(f'row bounds ({min_max_vals[0]}, {min_max_vals[1]}) '
                                                   f'are not sensible for NumRows {sicd.ImageData.NumRows}')):
        sicd.create_subset_structure(row_bounds=min_max_vals)
    min_max_vals = (100000, 100)
    with pytest.raises(ValueError, match=re.escape(f'column bounds ({min_max_vals[0]}, {min_max_vals[1]}) '
                                                   f'are not sensible for NumCols {sicd.ImageData.NumCols}')):
        sicd.create_subset_structure(column_bounds=min_max_vals)


def test_can_project_coordinates_geo1(sicd, caplog):
    sicd.GeoData.SCP = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because GeoData.SCP is not populated' in caplog.text


def test_can_project_coordinates_geo2(sicd, caplog):
    sicd.GeoData = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because GeoData is not populated' in caplog.text


def test_can_project_coordinates_image1(sicd, caplog):
    sicd.ImageData.SCPPixel = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because ImageData.SCPPixel is not populated' in caplog.text


def test_can_project_coordinates_image2(sicd, caplog):
    sicd.ImageData.FirstCol = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because ImageData.FirstCol is not populated' in caplog.text


def test_can_project_coordinates_image3(sicd, caplog):
    sicd.ImageData.FirstRow = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because ImageData.FirstRow is not populated' in caplog.text


def test_can_project_coordinates_image4(sicd, caplog):
    sicd.ImageData = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because ImageData is not populated' in caplog.text


def test_can_project_coordinates_pos1(sicd, caplog):
    sicd.Position.ARPPoly = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Position.ARPPoly is not populated' in caplog.text


def test_can_project_coordinates_pos2(sicd, caplog):
    sicd.Position = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Position is not populated' in caplog.text


def test_can_project_coordinates_grid1(sicd, caplog):
    sicd.Grid.Type = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Grid.Type is not populated' in caplog.text


def test_can_project_coordinates_grid2(sicd, caplog):
    sicd.Grid.Col.SS = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Grid.Col.SS is not populated' in caplog.text


def test_can_project_coordinates_grid3(sicd, caplog):
    sicd.Grid.Col = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Grid.Col is not populated' in caplog.text


def test_can_project_coordinates_grid4(sicd, caplog):
    sicd.Grid.Row.SS = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Grid.Row.SS is not populated' in caplog.text


def test_can_project_coordinates_grid5(sicd, caplog):
    sicd.Grid.Row = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Grid.Row is not populated' in caplog.text


def test_can_project_coordinates_grid6(sicd, caplog):
    sicd.Grid.TimeCOAPoly = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection may be inaccurate, because Grid.TimeCOAPoly is not populated' in caplog.text


def test_can_project_coordinates_grid7(sicd, caplog):
    sicd.Grid = None
    sicd.can_project_coordinates()
    assert 'Formulating a projection is not feasible because Grid is not populated' in caplog.text


def test_can_project_coordinates_grid8(sicd, caplog):
    sicd.Grid.Type = 'PLANE'
    sicd.Grid.Row.UVectECF = None
    sicd.can_project_coordinates()
    assert 'UVectECF parameter of Grid.Row or Grid.Col is not populated' in caplog.text


def test_can_project_coordinates_grid9(sicd, caplog):
    sicd.Grid.Type = 'BADTYPE'
    sicd.can_project_coordinates()
    assert 'Unhandled Grid.Type' in caplog.text


def test_can_project_coordinates_pfa1(sicd, caplog):
    sicd.PFA.SpatialFreqSFPoly = None
    sicd.can_project_coordinates()
    assert 'the PFA.SpatialFreqSFPoly parameter is not populated' in caplog.text


def test_can_project_coordinates_pfa2(sicd, caplog):
    sicd.PFA.PolarAngPoly = None
    sicd.can_project_coordinates()
    assert 'but the PFA.PolarAngPoly parameter is not populated' in caplog.text


def test_can_project_coordinates_pfa3(sicd, caplog):
    sicd.PFA = None
    sicd.can_project_coordinates()
    assert 'but the PFA parameter is not populated' in caplog.text


def test_can_project_coordinates_if1(sicd, caplog):
    sicd.Grid.Type = 'RGAZIM'
    sicd.ImageFormation.ImageFormAlgo = None
    sicd.can_project_coordinates()
    assert 'ImageFormation.ImageFormAlgo is not populated' in caplog.text


def test_can_project_coordinates_if2(sicd, caplog):
    sicd.ImageFormation = None
    sicd.can_project_coordinates()
    assert 'ImageFormation is not populated' in caplog.text


def test_can_project_coordinates_if3(sicd, caplog):
    sicd.ImageFormation.ImageFormAlgo = 'PFACOMP'
    sicd.can_project_coordinates()
    assert 'got unhandled ImageFormation.ImageFormAlgo' in caplog.text


def test_can_project_coordinates_rgazcomp(sicd, caplog):
    sicd.ImageFormation.ImageFormAlgo = 'RGAZCOMP'
    rg_az_comp_type = RgAzComp.RgAzCompType()
    rg_az_comp_type._derive_parameters(sicd.Grid, sicd.Timeline, sicd.SCPCOA)
    sicd.RgAzComp = rg_az_comp_type
    sicd.RgAzComp.AzSF = None
    sicd.can_project_coordinates()
    assert 'RgAzComp.AzSF parameter is not populated' in caplog.text
    sicd.RgAzComp = None
    sicd.can_project_coordinates()
    assert 'RgAzComp parameter is not populated' in caplog.text


def test_can_project_coordinates_rma(rma_sicd, caplog):
    bad_rma_sicd = copy.copy(rma_sicd)
    bad_rma_sicd.Grid.Type = 'RGZERO'
    bad_rma_sicd.RMA.INCA = None
    bad_rma_sicd.can_project_coordinates()
    assert 'but the RMA.INCA parameter is not populated' in caplog.text
