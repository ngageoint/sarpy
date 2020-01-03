from .. import generic_construction_test, unittest

from sarpy.sicd_elements import ImageFormation


rcv_chan_proc_dict = {'NumChanProc': 10, 'PRFScaleFactor': 1, 'ChanIndices': [1, 4], }
tx_freq_proc_dict = {'MinProc': 1, 'MaxProc': 10}
processing_dict = {'Type': 'Some Value', 'Applied': True, 'Parameters': {'Name': 'Value'}}
distortion_dict = {
    'CalibrationDate': '2019-12-10T19:40:31.000000Z',
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
        item1 = generic_construction_test(self, the_type, the_dict)


class TestTxFrequencyProc(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.TxFrequencyProcType
        the_dict = tx_freq_proc_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestProcessing(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.ProcessingType
        the_dict = processing_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestDistortion(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.DistortionType
        the_dict = distortion_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestPolarizationCalibration(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.PolarizationCalibrationType
        the_dict = polarization_calibration_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestImageFormation(unittest.TestCase):
    def test_construction(self):
        the_type = ImageFormation.ImageFormationType
        the_dict = image_formation_dict
        item1 = generic_construction_test(self, the_type, the_dict)
