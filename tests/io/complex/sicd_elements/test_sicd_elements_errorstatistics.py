#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.complex.sicd_elements import blocks
from sarpy.io.complex.sicd_elements import ErrorStatistics


def test_errorstatistics(kwargs):
    scp_error_type = ErrorStatistics.CompositeSCPErrorType(Rg=1.0, Az=2.0, RgAz=3.0, **kwargs)
    assert scp_error_type._xml_ns == kwargs['_xml_ns']
    assert scp_error_type._xml_ns_key == kwargs['_xml_ns_key']

    corr_coefs_type = ErrorStatistics.CorrCoefsType(P1P2=1.0, P1P3=2.0, P1V1=3.0, P1V2=4.0, P1V3=5.0,
                                                    P2P3=5.0, P2V1=4.0, P2V2=3.0, P2V3=2.0, P3V1=1.0,
                                                    P3V2=1.0, P3V3=2.0, V1V2=3.0, V1V3=4.0, V2V3=5.0,
                                                    **kwargs)
    assert corr_coefs_type._xml_ns == kwargs['_xml_ns']
    assert corr_coefs_type._xml_ns_key == kwargs['_xml_ns_key']

    pos_vel_err_type = ErrorStatistics.PosVelErrType(Frame='ECF', P1=1.0, P2=2.0, P3=3.0,
                                                     V1=3.0, V2=2.0, V3=1.0, **kwargs)
    assert pos_vel_err_type._xml_ns == kwargs['_xml_ns']
    assert pos_vel_err_type._xml_ns_key == kwargs['_xml_ns_key']

    radar_sensor_error_type =  ErrorStatistics.RadarSensorErrorType(RangeBias=1.0, **kwargs)
    assert radar_sensor_error_type._xml_ns == kwargs['_xml_ns']
    assert radar_sensor_error_type._xml_ns_key == kwargs['_xml_ns_key']

    tropo_error_type =  ErrorStatistics.TropoErrorType(**kwargs)
    assert tropo_error_type._xml_ns == kwargs['_xml_ns']
    assert tropo_error_type._xml_ns_key == kwargs['_xml_ns_key']

    iono_error_type =  ErrorStatistics.IonoErrorType(IonoRgRgRateCC=1.0, **kwargs)
    assert iono_error_type._xml_ns == kwargs['_xml_ns']
    assert iono_error_type._xml_ns_key == kwargs['_xml_ns_key']

    error_comp_type =  ErrorStatistics.ErrorComponentsType(PosVelErr=pos_vel_err_type,
                                                           RadarSensor=radar_sensor_error_type,
                                                           **kwargs)
    assert error_comp_type._xml_ns == kwargs['_xml_ns']
    assert error_comp_type._xml_ns_key == kwargs['_xml_ns_key']

    unmodeled_decorr_type =  ErrorStatistics.UnmodeledDecorrType(Xrow=blocks.ErrorDecorrFuncType(CorrCoefZero=0.0, DecorrRate=2.0),
                                                                 Ycol=blocks.ErrorDecorrFuncType(CorrCoefZero=0.0, DecorrRate=4.0),
                                                                 **kwargs)
    assert unmodeled_decorr_type._xml_ns == kwargs['_xml_ns']
    assert unmodeled_decorr_type._xml_ns_key == kwargs['_xml_ns_key']

    unmodeled_type =  ErrorStatistics.UnmodeledType(Xrow=1.0, Ycol=2.0, XrowYcol=3.0, **kwargs)
    assert unmodeled_type._xml_ns == kwargs['_xml_ns']
    assert unmodeled_type._xml_ns_key == kwargs['_xml_ns_key']

    error_stats_type =  ErrorStatistics.ErrorStatisticsType(**kwargs)
    assert error_stats_type._xml_ns == kwargs['_xml_ns']
    assert error_stats_type._xml_ns_key == kwargs['_xml_ns_key']

    assert error_stats_type.version_required() == (1, 1, 0)

    error_stats_type =  ErrorStatistics.ErrorStatisticsType(Unmodeled=unmodeled_type)
    assert error_stats_type.version_required() == (1, 3, 0)
