
from sarpy.io.complex.sicd_elements import ErrorStatistics

from . import generic_construction_test, unittest


scp_error_dict = {'Rg': 1, 'Az': 2, 'RgAz': 0.5}

corr_coefs_dict = {
    'P1P2': 0, 'P1P3': 0, 'P1V1': 0, 'P1V2': 0, 'P1V3': 0,
    'P2P3': 0, 'P2V1': 0, 'P2V2': 0, 'P2V3': 0,
    'P3V1': 0, 'P3V2': 0, 'P3V3': 0,
    'V1V2': 0, 'V1V3': 0,
    'V2V3': 0,
}

pos_vel_err_dict = {
    'Frame': 'ECF',
    'P1': 0, 'P2': 0, 'P3': 0,
    'V1': 0, 'V2': 0, 'V3': 0,
    'CorrCoefs': corr_coefs_dict,
    'PositionDecorr': {'CorrCoefZero': 0, 'DecorrRate': 0},
}

radar_sensor_error_dict = {
    'RangeBias': 0,
    'ClockFreqSF': 0,
    'TransmitFreqSF': 0,
    'RangeBiasDecorr': {'CorrCoefZero': 0, 'DecorrRate': 0},
}

tropo_error_dict = {
    'TropoRangeVertical': 0,
    'TropoRangeSlant': 0,
    'TropoRangeDecorr': {'CorrCoefZero': 0, 'DecorrRate': 0},
}

iono_error_dict = {
    'IonoRangeVertical': 0,
    'IonoRangeSlant': 0,
    'IonoRgRgRateCC': 0,
    'IonoRangeDecorr': {'CorrCoefZero': 0, 'DecorrRate': 0},
}

error_components_dict = {
    'PosVelErr': pos_vel_err_dict,
    'RadarSensor': radar_sensor_error_dict,
    'TropoError': tropo_error_dict,
    'IonoError': iono_error_dict,
}

error_statistics_dict = {
    'CompositeSCP': scp_error_dict,
    'Components': error_components_dict,
    'AdditionalParms': {'Name': 'Value'},
}


class TestCompositeSCPError(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.CompositeSCPErrorType
        the_dict = scp_error_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestCorrCoefs(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.CorrCoefsType
        the_dict = corr_coefs_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestPosVelError(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.PosVelErrType
        the_dict = pos_vel_err_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestRadarSensorError(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.RadarSensorErrorType
        the_dict = radar_sensor_error_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestTropoError(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.TropoErrorType
        the_dict = tropo_error_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestIonoError(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.IonoErrorType
        the_dict = iono_error_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestErrorComponents(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.ErrorComponentsType
        the_dict = error_components_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestErrorStatistics(unittest.TestCase):
    def test_construction(self):
        the_type = ErrorStatistics.ErrorStatisticsType
        the_dict = error_statistics_dict
        item1 = generic_construction_test(self, the_type, the_dict)
