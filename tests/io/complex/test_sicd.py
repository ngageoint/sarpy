import json
import numpy
import os
import pytest
import tempfile
import unittest

from sarpy.io.complex.converter import conversion_utility, open_complex
from sarpy.io.complex.sicd import SICDReader, SICDWriter, SICDWritingDetails
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_schema import get_schema_path, get_default_version_string


from tests import parse_file_entry

try:
    from lxml import etree
except ImportError:
    etree = None


complex_file_types = {}
this_loc = os.path.abspath(__file__)
file_reference = os.path.join(os.path.split(this_loc)[0], 'complex_file_types.json')  # specifies file locations
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as fi:
        the_files = json.load(fi)
        for the_type in the_files:
            valid_entries = []
            for entry in the_files[the_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            complex_file_types[the_type] = valid_entries

sicd_files = complex_file_types.get('SICD', [])

the_version = get_default_version_string()
the_schema = get_schema_path(the_version)

def test_sicd_writer_init_failure_no_input(tmp_path):
    with pytest.raises(TypeError, 
                       match="missing 1 required positional argument: 'file_object'"):
        sicd_writer = SICDWriter() 

def test_sicd_writer_init_failure_file_only(tmp_path):
    output_file = tmp_path / "out.sicd"
    with pytest.raises(ValueError, 
                       match="One of sicd_meta or sicd_writing_details must be provided."):
        sicd_writer = SICDWriter(output_file) 

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_meta_only(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    with pytest.raises(TypeError, 
                       match="missing 1 required positional argument: 'file_object'"):
        sicd_writer = SICDWriter(sicd_meta=sicd_meta)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_writing_details_only(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file = str(tmp_path / "out.sicd")
    with pytest.raises(TypeError, 
                       match="missing 1 required positional argument: 'file_object'"):
        sicd_writer = SICDWriter(sicd_writing_details=sicd_writing_details)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_meta(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_writing_details(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_writing_details=sicd_writing_details)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_meta_invalid_output(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    output_file = str("/does_not_exist/out.sicd")
    with pytest.raises(FileNotFoundError, 
                       match="No such file or directory: '/does_not_exist/out.sicd'"):
        sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_sicd_writing_details_invalid_output(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file = str("/does_not_exist/out.sicd")
    with pytest.raises(FileNotFoundError, 
                       match="No such file or directory: '/does_not_exist/out.sicd'"):
        sicd_writer = SICDWriter(output_file, sicd_writing_details=sicd_writing_details)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_bad_sicd_meta(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    sicd_meta_bad_string = str(sicd_meta) + "})"
    sicd_meta_bad_type = SICDType(sicd_meta_bad_string)
    output_file = str(tmp_path / "out.sicd")
    with pytest.raises(ValueError, 
                       match="The sicd_meta has un-populated ImageData, and nothing useful can be inferred."):
        sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta_bad_type)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_writer_init_bad_sicd_writing_details(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    # sicd_meta_bad_string = str(sicd_meta) + "})"
    # sicd_meta_bad_type = SICDType(sicd_meta_bad_string)
    sicd_writing_details_bad = "Bad Data"
    output_file = str(tmp_path / "out.sicd")
    with pytest.raises(TypeError, 
                       match="nitf_writing_details must be of type <class 'sarpy.io.complex.sicd.SICDWritingDetails'>"):
        sicd_writer = SICDWriter(output_file, sicd_writing_details=sicd_writing_details_bad)
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_nitf_writing_details(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_writing_details=sicd_writing_details)
    assert sicd_writer.nitf_writing_details == sicd_writing_details

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_nitf_writing_details_setter(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    sicd_writing_details = SICDWritingDetails(sicd_meta)
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_writing_details=sicd_writing_details)
    with pytest.raises(ValueError, 
                       match="nitf_writing_details is read-only"):
        sicd_writer.nitf_writing_details = sicd_writing_details

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_get_format_function(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)
    sicd_writer.get_format_function(raw_dtype="float", band_dimension=8, 
                                    complex_order='IQ', lut=numpy.array([1, 2]))


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_get_format_function_bad_band_type(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)
    with pytest.raises(ValueError, 
                       match="Got unsupported SICD band type definition"):
        sicd_writer.get_format_function(raw_dtype="float", band_dimension=8, 
                                    complex_order='Steve', lut=numpy.array([1, 2]))


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicd_meta(tmp_path):
    input_file = sicd_files[0]
    reader = open_complex(input_file)
    sicd_meta = reader.sicd_meta
    output_file = str(tmp_path / "out.sicd")
    sicd_writer = SICDWriter(output_file, sicd_meta=sicd_meta)
    # Must confirm each piece of the SICDType is equal because there isn't an
    # == operator for SICDType.
    # First confirm that all CollectionInfo properties are equal.
    assert sicd_meta.CollectionInfo.Classification == \
        sicd_writer.sicd_meta.CollectionInfo.Classification
    assert sicd_meta.CollectionInfo.CollectorName == \
        sicd_writer.sicd_meta.CollectionInfo.CollectorName
    assert sicd_meta.CollectionInfo.IlluminatorName == \
        sicd_writer.sicd_meta.CollectionInfo.IlluminatorName
    assert sicd_meta.CollectionInfo.CoreName == \
        sicd_writer.sicd_meta.CollectionInfo.CoreName
    assert sicd_meta.CollectionInfo.CollectType == \
        sicd_writer.sicd_meta.CollectionInfo.CollectType
    assert sicd_meta.CollectionInfo.CountryCodes == \
        sicd_writer.sicd_meta.CollectionInfo.CountryCodes
    assert sicd_meta.CollectionInfo.Parameters == \
        sicd_writer.sicd_meta.CollectionInfo.Parameters
    assert sicd_meta.CollectionInfo.RadarMode.ModeType == \
        sicd_writer.sicd_meta.CollectionInfo.RadarMode.ModeType
    assert sicd_meta.CollectionInfo.RadarMode.ModeID == \
        sicd_writer.sicd_meta.CollectionInfo.RadarMode.ModeID
    assert sicd_meta.ImageCreation.Application == \
        sicd_writer.sicd_meta.ImageCreation.Application
    assert sicd_meta.ImageCreation.DateTime == \
        sicd_writer.sicd_meta.ImageCreation.DateTime
    assert sicd_meta.ImageCreation.Site == \
        sicd_writer.sicd_meta.ImageCreation.Site
    # TODO: These are different because one is based on the profile from the
    #       file and the other is based on the profile of the version of sarpy
    #       we are testing. Skip for now. Determine if these should be the same
    # assert sicd_meta.ImageCreation.Profile == \
    #     sicd_writer.sicd_meta.ImageCreation.Profile
    assert sicd_meta.ImageData.AmpTable == \
        sicd_writer.sicd_meta.ImageData.AmpTable
    assert sicd_meta.ImageData.FirstCol == \
        sicd_writer.sicd_meta.ImageData.FirstCol
    assert sicd_meta.ImageData.FirstRow == \
        sicd_writer.sicd_meta.ImageData.FirstRow
    assert sicd_meta.ImageData.PixelType == \
        sicd_writer.sicd_meta.ImageData.PixelType
    assert sicd_meta.ImageData.SCPPixel.Row == \
        sicd_writer.sicd_meta.ImageData.SCPPixel.Row
    assert sicd_meta.ImageData.SCPPixel.Col == \
        sicd_writer.sicd_meta.ImageData.SCPPixel.Col
    assert sicd_meta.ImageData.FullImage.NumCols == \
        sicd_writer.sicd_meta.ImageData.FullImage.NumCols
    assert sicd_meta.ImageData.FullImage.NumRows == \
        sicd_writer.sicd_meta.ImageData.FullImage.NumRows
    assert sicd_meta.GeoData.EarthModel == \
        sicd_writer.sicd_meta.GeoData.EarthModel
    assert sicd_meta.GeoData.SCP.ECF.X == \
        sicd_writer.sicd_meta.GeoData.SCP.ECF.X
    assert sicd_meta.GeoData.SCP.ECF.Y == \
        sicd_writer.sicd_meta.GeoData.SCP.ECF.Y
    assert sicd_meta.GeoData.SCP.ECF.Z == \
        sicd_writer.sicd_meta.GeoData.SCP.ECF.Z
    assert sicd_meta.GeoData.SCP.LLH.Lat == \
        sicd_writer.sicd_meta.GeoData.SCP.LLH.Lat
    assert sicd_meta.GeoData.SCP.LLH.Lon == \
        sicd_writer.sicd_meta.GeoData.SCP.LLH.Lon
    assert sicd_meta.GeoData.SCP.LLH.HAE == \
        sicd_writer.sicd_meta.GeoData.SCP.LLH.HAE
    # TODO: Confirm ImageCorners match.
    # assert sicd_meta.GeoData.ImageCorners.FRFC == \
    #     sicd_writer.sicd_meta.GeoData.ImageCorners.FRFC
    # assert sicd_meta.GeoData.ImageCorners.FRLC == \
    #     sicd_writer.sicd_meta.GeoData.ImageCorners.FRLC
    # assert sicd_meta.GeoData.ImageCorners.LRFC == \
    #     sicd_writer.sicd_meta.GeoData.ImageCorners.LRFC
    # assert sicd_meta.GeoData.ImageCorners.LRLC == \
    #     sicd_writer.sicd_meta.GeoData.ImageCorners.LRLC
    assert sicd_meta.Grid.Col.SS == \
        sicd_writer.sicd_meta.Grid.Col.SS
    assert sicd_meta.Grid.Col.UVectECF.X == \
        sicd_writer.sicd_meta.Grid.Col.UVectECF.X
    assert sicd_meta.Grid.Col.UVectECF.Y == \
        sicd_writer.sicd_meta.Grid.Col.UVectECF.Y
    assert sicd_meta.Grid.Col.UVectECF.Z == \
        sicd_writer.sicd_meta.Grid.Col.UVectECF.Z
    assert sicd_meta.Grid.Row.SS == \
        sicd_writer.sicd_meta.Grid.Row.SS
    assert sicd_meta.Grid.Row.UVectECF.X == \
        sicd_writer.sicd_meta.Grid.Row.UVectECF.X
    assert sicd_meta.Grid.Row.UVectECF.Y == \
        sicd_writer.sicd_meta.Grid.Row.UVectECF.Y
    assert sicd_meta.Grid.Row.UVectECF.Z == \
        sicd_writer.sicd_meta.Grid.Row.UVectECF.Z
    # assert sicd_meta.Grid == \
    #     sicd_writer.sicd_meta.Grid
    
    #  'Grid', 'Timeline', 'Position',
    #     'RadarCollection', 'ImageFormation', 'SCPCOA', 'Radiometric', 'Antenna', 'ErrorStatistics',
    #     'MatchInfo', 'RgAzComp', 'PFA', 'RMA'

class TestSICDWriting(unittest.TestCase):

    def setUp(self):
        self.sicd_files = complex_file_types.get('SICD', [])
        if len(self.sicd_files) == 0 :
            self.skipTest('No sicd files found')
    
    def test_sicd_creation(self):
        for fil in self.sicd_files:
            reader = SICDReader(fil)

            # check that sicd structure serializes according to the schema
            if etree is not None:
                sicd = reader.get_sicds_as_tuple()[0]
                xml_doc = etree.fromstring(sicd.to_xml_bytes())
                xml_schema = etree.XMLSchema(file=the_schema)
                with self.subTest(msg='validate xml produced from sicd structure'):
                    self.assertTrue(xml_schema.validate(xml_doc),
                                    msg='SICD structure serialized from file {} is '
                                        'not valid versus schema {}'.format(fil, the_schema))

            with self.subTest(msg='Test conversion (recreation) of the sicd file {}'.format(fil)):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    conversion_utility(reader, tmpdirname)
                    new_filename = os.path.join(tmpdirname, os.listdir(tmpdirname)[0])
                    with SICDReader(new_filename) as reader2:
                        self.assertEqual(os.stat(new_filename).st_size, reader2.nitf_details.nitf_header.FL)

            with self.subTest(msg='Test writing a single row of the sicd file {}'.format(fil)):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    conversion_utility(reader, tmpdirname, row_limits=(0, 1))
