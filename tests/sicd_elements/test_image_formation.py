import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import ImageFormation


rcv_chan_proc_dict = {'NumChanProc': 10, 'PRFScaleFactor': 1, 'ChanIndices': [1, 4], }
tx_freq_proc_dict = {'MinProc': 1, 'MaxProc': 10}
processing_dict = {'Type': 'Some Value', 'Applied': True, 'Parameters': {'Name': 'Value'}}
distortion_dict = {
    'CalibrationDate': '2019-12-10T19:40:31.000000',
    'A': 1,
    'F1': {'Real': 1, 'Imag': 1},
    'Q1': {'Real': 1, 'Imag': 1},
    'Q2': {'Real': 1, 'Imag': 1},
    'F2': {'Real': 1, 'Imag': 1},
    'Q3': {'Real': 1, 'Imag': 1},
    'Q4': {'Real': 1, 'Imag': 1},
    'GainErrorA': 1,
    'GainErrorF1': 1,
    'GainErrorF2': 1,
    'PhaseErrorF1': 1,
    'PhaseErrorF2': 1,
}
polarization_calibration_dict = {'DistortCorrectApplied': True, 'Distortion': distortion_dict}
image_formation_dict = {
    'RcvChanProc': rcv_chan_proc_dict,
    'TxRcvPolarizationProc': 'V:V',
    'TStartProc': 1,
    'TEndProc': 10,
    'TxFrequencyProc': tx_freq_proc_dict,
    'SegmentIdentifier': 'the_id',
    'ImageFormAlgo': 'OTHER',
    'STBeamComp': 'NO',
    'ImageBeamComp': 'NO',
    'AzAutofocus': 'NO',
    'RgAutofocus': 'NO',
    'Processings': [processing_dict, ],
    'PolarizationCalibration': polarization_calibration_dict,
}


class TestRcvChanProc(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.RcvChanProcType
        the_dict = rcv_chan_proc_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())


class TestTxFrequencyProc(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.TxFrequencyProcType
        the_dict = tx_freq_proc_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())


class TestProcessing(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.ProcessingType
        the_dict = processing_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())


class TestDistortion(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.DistortionType
        the_dict = distortion_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())


class TestPolarizationCalibration(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.PolarizationCalibrationType
        the_dict = polarization_calibration_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())


class TestImageFormation(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.ImageFormationType
        the_dict = image_formation_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())
