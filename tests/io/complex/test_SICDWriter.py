import json
import numpy
import os
import pytest
import unittest

from sarpy.io.complex.converter          import conversion_utility, open_complex
from sarpy.io.complex.sicd               import SICDWriter, SICDWritingDetails
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_schema        import get_schema_path, \
    get_default_version_string
from sarpy.io.complex.sicd_elements.blocks import XYZPolyType

from tests import parse_file_entry

complex_file_types = {}
this_loc           = os.path.abspath(__file__)
# specifies file locations
file_reference     = os.path.join(os.path.split(this_loc)[0], \
                                  'complex_file_types.json')  
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as local_file:
        test_files_list = json.load(local_file)
        for test_files_type in test_files_list:
            valid_entries = []
            for entry in test_files_list[test_files_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            complex_file_types[test_files_type] = valid_entries

sicd_files = complex_file_types.get('SICD', [])

def get_sicd_meta():
    input_file       = sicd_files[0]
    reader           = open_complex(input_file)
    return_sicd_meta = reader.sicd_meta
    return return_sicd_meta

def test_sicd_writer_init_failure_no_input(tmp_path):
    with pytest.raises(TypeError, 
                       match="missing 1 required positional argument: " + \
                        "'file_object'"):
        sicd_writer = SICDWriter() 

def test_sicd_writer_init_failure_file_only(tmp_path):
    output_file = tmp_path / "out.sicd"
    with pytest.raises(ValueError, 
                       match="One of sicd_meta or sicd_writing_details must " + \
                        "be provided."):
        sicd_writer = SICDWriter(output_file) 

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_failure_sicd_meta_only(tmp_path):
    sicd_meta = get_sicd_meta()
    with pytest.raises(TypeError, 
                       match="missing 1 required positional argument: " + \
                        "'file_object'"):
        sicd_writer = SICDWriter(sicd_meta=sicd_meta)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_failure_sicd_writing_details_only(tmp_path):
    sicd_meta            = get_sicd_meta()
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file          = str(tmp_path / "out.sicd")
    with pytest.raises(TypeError, 
                       match="missing 1 required positional argument: " + \
                        "'file_object'"):
        sicd_writer = SICDWriter(sicd_writing_details=sicd_writing_details)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_meta(tmp_path):
    sicd_meta   = get_sicd_meta()
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_writing_details(tmp_path):
    sicd_meta            = get_sicd_meta()
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file          = str(tmp_path / "out.sicd")
    sicd_writer          = SICDWriter(output_file, \
                             sicd_writing_details=sicd_writing_details)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_failure_sicd_meta_invalid_output(tmp_path):
    sicd_meta   = get_sicd_meta()
    output_file = str("/does_not_exist/out.sicd")
    with pytest.raises(FileNotFoundError, 
                       match="No such file or directory: " + \
                        "'/does_not_exist/out.sicd'"):
        sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_failure_sicd_writing_details_invalid_output(tmp_path):
    sicd_meta            = get_sicd_meta()
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file          = str("/does_not_exist/out.sicd")
    with pytest.raises(FileNotFoundError, 
                       match="No such file or directory: " + \
                        "'/does_not_exist/out.sicd'"):
        sicd_writer = SICDWriter(output_file, \
                                 sicd_writing_details=sicd_writing_details)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_failure_bad_sicd_meta(tmp_path):
    sicd_meta            = get_sicd_meta()
    sicd_meta_bad_string = str(sicd_meta) + "})"
    sicd_meta_bad_type   = SICDType(sicd_meta_bad_string)
    output_file          = str(tmp_path / "out.sicd")
    with pytest.raises(ValueError, 
                       match="The sicd_meta has un-populated ImageData, and " + \
                        "nothing useful can be inferred."):
        sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta_bad_type)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_failure_bad_sicd_writing_details(tmp_path):
    sicd_writing_details_bad = "Bad Data"
    output_file              = str(tmp_path / "out.sicd")
    with pytest.raises(TypeError, 
                       match="nitf_writing_details must be of type " + \
                        "<class 'sarpy.io.complex.sicd.SICDWritingDetails'>"):
        sicd_writer = SICDWriter(output_file, \
                                 sicd_writing_details=sicd_writing_details_bad)
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_nitf_writing_details(tmp_path):
    sicd_meta            = get_sicd_meta()
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file          = str(tmp_path / "out.sicd")
    sicd_writer          = SICDWriter(output_file, \
                             sicd_writing_details=sicd_writing_details)
    assert sicd_writer.nitf_writing_details == sicd_writing_details

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_nitf_writing_details_setter_failure(tmp_path):
    sicd_meta            = get_sicd_meta()
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file          = str(tmp_path / "out.sicd")
    sicd_writer          = SICDWriter(output_file, \
                             sicd_writing_details=sicd_writing_details)
    with pytest.raises(ValueError, 
                       match="nitf_writing_details is read-only"):
        sicd_writer.nitf_writing_details = sicd_writing_details

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_get_format_function(tmp_path):
    sicd_meta   = get_sicd_meta()
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)
    sicd_writer.get_format_function(raw_dtype="float", band_dimension=8, 
                                    complex_order='IQ', lut=numpy.array([1, 2]))


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_get_format_function_failure_bad_band_type(tmp_path):
    sicd_meta   = get_sicd_meta()
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)
    with pytest.raises(ValueError, 
                       match="Got unsupported SICD band type definition"):
        sicd_writer.get_format_function(raw_dtype="float", band_dimension=8, 
                                    complex_order='Steve', \
                                        lut=numpy.array([1, 2]))


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_meta(tmp_path):
    sicd_meta   = get_sicd_meta()
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)
    # Must confirm each piece of the SICDType is equal because there isn't an
    # == operator for SICDType.
    # First confirm that all CollectionInfo properties are equal.
    assert sicd_meta.CollectionInfo.Classification        == \
        sicd_writer.sicd_meta.CollectionInfo.Classification
    assert sicd_meta.CollectionInfo.CollectorName         == \
        sicd_writer.sicd_meta.CollectionInfo.CollectorName
    assert sicd_meta.CollectionInfo.IlluminatorName       == \
        sicd_writer.sicd_meta.CollectionInfo.IlluminatorName
    assert sicd_meta.CollectionInfo.CoreName              == \
        sicd_writer.sicd_meta.CollectionInfo.CoreName
    assert sicd_meta.CollectionInfo.CollectType           == \
        sicd_writer.sicd_meta.CollectionInfo.CollectType
    assert sicd_meta.CollectionInfo.CountryCodes          == \
        sicd_writer.sicd_meta.CollectionInfo.CountryCodes
    assert sicd_meta.CollectionInfo.Parameters            == \
        sicd_writer.sicd_meta.CollectionInfo.Parameters
    assert sicd_meta.CollectionInfo.RadarMode.ModeType    == \
        sicd_writer.sicd_meta.CollectionInfo.RadarMode.ModeType
    assert sicd_meta.CollectionInfo.RadarMode.ModeID      == \
        sicd_writer.sicd_meta.CollectionInfo.RadarMode.ModeID
    # Confirm ImageCreation properties are equal.
    assert sicd_meta.ImageCreation.Application            == \
        sicd_writer.sicd_meta.ImageCreation.Application
    assert sicd_meta.ImageCreation.DateTime               == \
        sicd_writer.sicd_meta.ImageCreation.DateTime
    assert sicd_meta.ImageCreation.Site                   == \
        sicd_writer.sicd_meta.ImageCreation.Site
    # TODO: These are different because one is based on the profile from the
    #       file and the other is based on the profile of the version of sarpy
    #       we are testing. Skip for now. Determine if these should be the same
    # assert sicd_meta.ImageCreation.Profile == \
    #     sicd_writer.sicd_meta.ImageCreation.Profile
    # Confirm ImageData properties are equal.
    assert sicd_meta.ImageData.AmpTable                   == \
        sicd_writer.sicd_meta.ImageData.AmpTable
    assert sicd_meta.ImageData.FirstCol                   == \
        sicd_writer.sicd_meta.ImageData.FirstCol
    assert sicd_meta.ImageData.FirstRow                   == \
        sicd_writer.sicd_meta.ImageData.FirstRow
    assert sicd_meta.ImageData.PixelType                  == \
        sicd_writer.sicd_meta.ImageData.PixelType
    assert sicd_meta.ImageData.SCPPixel.Row               == \
        sicd_writer.sicd_meta.ImageData.SCPPixel.Row
    assert sicd_meta.ImageData.SCPPixel.Col               == \
        sicd_writer.sicd_meta.ImageData.SCPPixel.Col
    assert sicd_meta.ImageData.FullImage.NumCols          == \
        sicd_writer.sicd_meta.ImageData.FullImage.NumCols
    assert sicd_meta.ImageData.FullImage.NumRows          == \
        sicd_writer.sicd_meta.ImageData.FullImage.NumRows
    # Confirm GeoData properties are equal.
    assert sicd_meta.GeoData.EarthModel                   == \
        sicd_writer.sicd_meta.GeoData.EarthModel
    assert sicd_meta.GeoData.SCP.ECF.X                    == \
        sicd_writer.sicd_meta.GeoData.SCP.ECF.X
    assert sicd_meta.GeoData.SCP.ECF.Y                    == \
        sicd_writer.sicd_meta.GeoData.SCP.ECF.Y
    assert sicd_meta.GeoData.SCP.ECF.Z                    == \
        sicd_writer.sicd_meta.GeoData.SCP.ECF.Z
    assert sicd_meta.GeoData.SCP.LLH.Lat                  == \
        sicd_writer.sicd_meta.GeoData.SCP.LLH.Lat
    assert sicd_meta.GeoData.SCP.LLH.Lon                  == \
        sicd_writer.sicd_meta.GeoData.SCP.LLH.Lon
    assert sicd_meta.GeoData.SCP.LLH.HAE                  == \
        sicd_writer.sicd_meta.GeoData.SCP.LLH.HAE
    # TODO: Confirm ImageCorners match.
    assert numpy.allclose(sicd_meta.GeoData.ImageCorners.FRFC,  \
        sicd_writer.sicd_meta.GeoData.ImageCorners.FRFC)
    assert numpy.allclose(sicd_meta.GeoData.ImageCorners.FRLC, \
        sicd_writer.sicd_meta.GeoData.ImageCorners.FRLC)
    assert numpy.allclose(sicd_meta.GeoData.ImageCorners.LRFC, \
        sicd_writer.sicd_meta.GeoData.ImageCorners.LRFC)
    assert numpy.allclose(sicd_meta.GeoData.ImageCorners.LRLC, \
        sicd_writer.sicd_meta.GeoData.ImageCorners.LRLC)
    # Confirm Grid properties are equal.
    assert sicd_meta.Grid.Col.SS                          == \
        sicd_writer.sicd_meta.Grid.Col.SS
    assert sicd_meta.Grid.Col.UVectECF.X                  == \
        sicd_writer.sicd_meta.Grid.Col.UVectECF.X
    assert sicd_meta.Grid.Col.UVectECF.Y                  == \
        sicd_writer.sicd_meta.Grid.Col.UVectECF.Y
    assert sicd_meta.Grid.Col.UVectECF.Z                  == \
        sicd_writer.sicd_meta.Grid.Col.UVectECF.Z
    assert sicd_meta.Grid.Row.SS                          == \
        sicd_writer.sicd_meta.Grid.Row.SS
    assert pytest.approx(sicd_meta.Grid.Row.UVectECF.X)   == \
        pytest.approx(sicd_writer.sicd_meta.Grid.Row.UVectECF.X)
    assert pytest.approx(sicd_meta.Grid.Row.UVectECF.Y)   == \
        pytest.approx(sicd_writer.sicd_meta.Grid.Row.UVectECF.Y)
    assert pytest.approx(sicd_meta.Grid.Row.UVectECF.Z)   == \
        pytest.approx(sicd_writer.sicd_meta.Grid.Row.UVectECF.Z)
    assert sicd_meta.Grid.TimeCOAPoly.Coefs               == \
        sicd_writer.sicd_meta.Grid.TimeCOAPoly.Coefs
    assert sicd_meta.Grid.TimeCOAPoly.order1              == \
        sicd_writer.sicd_meta.Grid.TimeCOAPoly.order1
    assert sicd_meta.Grid.TimeCOAPoly.order2              == \
        sicd_writer.sicd_meta.Grid.TimeCOAPoly.order2
    assert sicd_meta.Grid.Type                            == \
        sicd_writer.sicd_meta.Grid.Type
    assert sicd_meta.Grid.ImagePlane                      == \
        sicd_writer.sicd_meta.Grid.ImagePlane
    # Confirm Timeline properties are equal.
    assert sicd_meta.Timeline.IPP                         == \
        sicd_writer.sicd_meta.Timeline.IPP
    assert sicd_meta.Timeline.CollectStart                == \
        sicd_writer.sicd_meta.Timeline.CollectStart
    assert sicd_meta.Timeline.CollectDuration             == \
        sicd_writer.sicd_meta.Timeline.CollectDuration
    # Confirm Position properties are equal.
    assert pytest.approx(sicd_meta.Position.ARPPoly.X)    == \
        pytest.approx(sicd_writer.sicd_meta.Position.ARPPoly.X)
    assert pytest.approx(sicd_meta.Position.ARPPoly.Y)    == \
        pytest.approx(sicd_writer.sicd_meta.Position.ARPPoly.Y)
    assert pytest.approx(sicd_meta.Position.ARPPoly.Z)    == \
        pytest.approx(sicd_writer.sicd_meta.Position.ARPPoly.Z)
    assert pytest.approx(sicd_meta.Position.GRPPoly.X)    == \
        pytest.approx(sicd_writer.sicd_meta.Position.GRPPoly.X)
    assert pytest.approx(sicd_meta.Position.GRPPoly.Y)    == \
        pytest.approx(sicd_writer.sicd_meta.Position.GRPPoly.Y)
    assert pytest.approx(sicd_meta.Position.GRPPoly.Z)    == \
        pytest.approx(sicd_writer.sicd_meta.Position.GRPPoly.Z)
    # Confirm TxAPCPoly properties are equal.
    if isinstance(sicd_meta.Position.TxAPCPoly, XYZPolyType) \
        and isinstance(sicd_meta.Position.TxAPCPoly, XYZPolyType):
            assert pytest.approx(sicd_meta.Position.TxAPCPoly.X) == \
                pytest.approx(sicd_writer.sicd_meta.Position.TxAPCPoly.X)
            assert pytest.approx(sicd_meta.Position.TxAPCPoly.Y) == \
                pytest.approx(sicd_writer.sicd_meta.Position.TxAPCPoly.Y)
            assert pytest.approx(sicd_meta.Position.TxAPCPoly.Z) == \
                pytest.approx(sicd_writer.sicd_meta.Position.TxAPCPoly.Z)
    # Confirm RcvAPC properties are equal.
    if isinstance(sicd_meta.Position.RcvAPC, XYZPolyType) \
        and isinstance(sicd_meta.Position.RcvAPC, XYZPolyType):
        assert pytest.approx(sicd_meta.Position.RcvAPC.X)   == \
            pytest.approx(sicd_writer.sicd_meta.Position.RcvAPC.X)
        assert pytest.approx(sicd_meta.Position.RcvAPC.Y)   == \
            pytest.approx(sicd_writer.sicd_meta.Position.RcvAPC.Y)
        assert pytest.approx(sicd_meta.Position.RcvAPC.Z)   == \
            pytest.approx(sicd_writer.sicd_meta.Position.RcvAPC.Z)
    # Confirm RadarCollection properties are equal.
    assert numpy.allclose(sicd_meta.RadarCollection.TxFrequency.get_array(), \
        sicd_writer.sicd_meta.RadarCollection.TxFrequency.get_array())
    assert sicd_meta.RadarCollection.RefFreqIndex         == \
        sicd_writer.sicd_meta.RadarCollection.RefFreqIndex
    # TODO: Make this comparison work.
    # assert numpy.allclose(sicd_meta.RadarCollection.Waveform.get_array(),  \
    #     sicd_writer.sicd_meta.RadarCollection.Waveform.get_array())
    assert sicd_meta.RadarCollection.TxPolarization       == \
        sicd_writer.sicd_meta.RadarCollection.TxPolarization
    
    # TODO: Make tests for ImageFormation. This type requires an in depth 
    # comparison
    # assert sicd_meta.ImageFormation.TxPolarization      == \
    #     sicd_writer.sicd_meta.ImageFormation.TxPolarization
    # Confirm SCPCOA properties are equal.
    assert pytest.approx(sicd_meta.SCPCOA.SCPTime)        == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.SCPTime)
    assert pytest.approx(sicd_meta.SCPCOA.ARPPos.X)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPPos.X)
    assert pytest.approx(sicd_meta.SCPCOA.ARPPos.Y)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPPos.Y)
    assert pytest.approx(sicd_meta.SCPCOA.ARPPos.Z)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPPos.Z)
    assert pytest.approx(sicd_meta.SCPCOA.ARPVel.X)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPVel.X)
    assert pytest.approx(sicd_meta.SCPCOA.ARPVel.Y)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPVel.Y)
    assert pytest.approx(sicd_meta.SCPCOA.ARPVel.Z)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPVel.Z)
    assert pytest.approx(sicd_meta.SCPCOA.ARPAcc.X)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPAcc.X)
    assert pytest.approx(sicd_meta.SCPCOA.ARPAcc.Y)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPAcc.Y)
    assert pytest.approx(sicd_meta.SCPCOA.ARPAcc.Z)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.ARPAcc.Z)
    assert sicd_meta.SCPCOA.SideOfTrack                   == \
        sicd_writer.sicd_meta.SCPCOA.SideOfTrack
    assert pytest.approx(sicd_meta.SCPCOA.SlantRange)     == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.SlantRange)
    assert pytest.approx(sicd_meta.SCPCOA.GroundRange)    == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.GroundRange)
    assert pytest.approx(sicd_meta.SCPCOA.DopplerConeAng) == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.DopplerConeAng)
    assert pytest.approx(sicd_meta.SCPCOA.GrazeAng)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.GrazeAng)
    assert pytest.approx(sicd_meta.SCPCOA.IncidenceAng)   == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.IncidenceAng)
    assert pytest.approx(sicd_meta.SCPCOA.TwistAng)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.TwistAng)
    assert pytest.approx(sicd_meta.SCPCOA.SlopeAng)       == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.SlopeAng)
    assert pytest.approx(sicd_meta.SCPCOA.AzimAng)        == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.AzimAng)
    assert pytest.approx(sicd_meta.SCPCOA.LayoverAng)     == \
        pytest.approx(sicd_writer.sicd_meta.SCPCOA.LayoverAng)
    
    # The following sicd_meta features are optional, so skip them in the test for
    #   now.
    #   'Radiometric', 'Antenna', 'ErrorStatistics', 'MatchInfo', 'RgAzComp', 
    #   'PFA', 'RMA'
