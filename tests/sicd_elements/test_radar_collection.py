from .. import generic_construction_test, unittest

from sarpy.sicd_elements import RadarCollection


tx_frequency_dict = {'Min': 1, 'Max': 2}
waveform_parameters_dict = {
    'TxPulseLength': 1,
    'TxRFBandwidth': 1,
    'TxFreqStart': 1,
    'TxFMRate': 1,
    'RcvDemodType': 'CHIRP',
    'RcvWindowLength': 1,
    'ADCSampleRate': 100,
    'RcvIFBandwidth': 1,
    'RcvFreqStart': 1,
    'RcvFMRate': 0,
    'index': 0,
}
tx_step_dict = {'WFIndex': 0, 'TxPolarization': 'V', 'index': 0}
chan_parameters_dict = {'TxRcvPolarization': 'V:V', 'RcvAPCIndex': 0, 'index': 0}
reference_point_dict = {'ECF': {'X': 0, 'Y': 0, 'Z': 0}, 'Line': 0.5, 'Sample': 0.5, 'name': 'Name'}
x_direction_dict = {'UVectECF': {'X': 1, 'Y': 0, 'Z': 0}, 'LineSpacing': 1, 'NumLines': 10, 'FirstLine': 2}
y_direction_dict = {'UVectECF': {'X': 1, 'Y': 0, 'Z': 0}, 'SampleSpacing': 1, 'NumSamples': 10, 'FirstSample': 2}
segment_array_dict = {
    'StartLine': 1,
    'StartSample': 1,
    'EndLine': 2,
    'EndSample': 2,
    'Identifier': 'the_id',
    'index': 0,
}
reference_plane_dict = {
    'RefPt': reference_point_dict,
    'XDir': x_direction_dict,
    'YDir': y_direction_dict,
    'SegmentList': [segment_array_dict, ],
    'Orientation': 'UP'
}
area_dict = {
    'Corner': [
        {'Lat': 0, 'Lon': 0, 'HAE': 0, 'index': 1},
        {'Lat': 0, 'Lon': 1, 'HAE': 0, 'index': 2},
        {'Lat': 1, 'Lon': 1, 'HAE': 0, 'index': 3},
        {'Lat': 1, 'Lon': 0, 'HAE': 0, 'index': 4},
    ],
    'Plane': reference_plane_dict,
}
radar_collection_dict = {
    'TxFrequency': tx_frequency_dict,
    'RefFreqIndex': 0,
    'Waveform': [waveform_parameters_dict, ],
    'TxPolarization': 'V',
    'TxSequence': [tx_step_dict, ],
    'RcvChannels': [chan_parameters_dict, ],
    'Area': area_dict,
    'Parameters': {'name': 'value'},
}


class TestTxFrequency(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.TxFrequencyType
        the_dict = tx_frequency_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestWaveformParameters(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.WaveformParametersType
        the_dict = waveform_parameters_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestTxStep(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.TxStepType
        the_dict = tx_step_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestChanParameters(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.ChanParametersType
        the_dict = chan_parameters_dict
        item1 = generic_construction_test(self, the_type, the_dict)

        with self.subTest(msg="get_transmit_polarization"):
            self.assertEqual(item1.get_transmit_polarization(), 'V')


class TestReferencePoint(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.ReferencePointType
        the_dict = reference_point_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestXDirection(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.XDirectionType
        the_dict = x_direction_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestYDirection(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.YDirectionType
        the_dict = y_direction_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestSegmentArray(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.SegmentArrayElement
        the_dict = segment_array_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestReferencePlane(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.ReferencePlaneType
        the_dict = reference_plane_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestArea(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.AreaType
        the_dict = area_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestRadarCollection(unittest.TestCase):
    def test_construction(self):
        the_type = RadarCollection.RadarCollectionType
        the_dict = radar_collection_dict
        item1 = generic_construction_test(self, the_type, the_dict)
