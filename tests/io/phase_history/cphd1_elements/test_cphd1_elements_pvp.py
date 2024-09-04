#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np

from sarpy.io.phase_history.cphd1_elements import PVP


def test_pvp_pvptype():
    """Test PVPType class"""
    pvp_type = PVP.PVPType(
        TxTime=PVP.PerVectorParameterF8(0),
        TxPos=PVP.PerVectorParameterXYZ(1),
        TxVel=PVP.PerVectorParameterXYZ(4),
        RcvTime=PVP.PerVectorParameterF8(7),
        RcvPos=PVP.PerVectorParameterXYZ(8),
        RcvVel=PVP.PerVectorParameterXYZ(11),
        SRPPos=PVP.PerVectorParameterXYZ(14),
        AmpSF=None,
        aFDOP=PVP.PerVectorParameterF8(17),
        aFRR1=PVP.PerVectorParameterF8(18),
        aFRR2=PVP.PerVectorParameterF8(19),
        FX1=PVP.PerVectorParameterF8(20),
        FX2=PVP.PerVectorParameterF8(21),
        FXN1=PVP.PerVectorParameterF8(22),
        FXN2=PVP.PerVectorParameterF8(23),
        TOA1=PVP.PerVectorParameterF8(24),
        TOA2=PVP.PerVectorParameterF8(25),
        TOAE1=PVP.PerVectorParameterF8(26),
        TOAE2=PVP.PerVectorParameterF8(27),
        TDTropoSRP=PVP.PerVectorParameterF8(28),
        TDIonoSRP=PVP.PerVectorParameterF8(29),
        SC0=PVP.PerVectorParameterF8(30),
        SCSS=PVP.PerVectorParameterF8(31),
        SIGNAL=PVP.PerVectorParameterI8(32),
        TxAntenna=PVP.TxAntennaType(TxACX=PVP.PerVectorParameterXYZ(33),
                                    TxACY=PVP.PerVectorParameterXYZ(36),
                                    TxEB=PVP.PerVectorParameterEB(39)),
        RcvAntenna=PVP.RcvAntennaType(RcvACX=PVP.PerVectorParameterXYZ(41),
                                      RcvACY=PVP.PerVectorParameterXYZ(44),
                                      RcvEB=PVP.PerVectorParameterEB(47)),
        AddedPVP=[PVP.UserDefinedPVPType(Name='userpvp', Offset=49, Size=1, Format='CI16')]
    )

    # The number of PVPs specified above is 30 (AmpSF is left out)
    assert len(pvp_type.get_vector_dtype()) == 30

    assert pvp_type.get_size() == 400

    assert pvp_type.get_offset_size_format('TxPos') == (8, 24, 'd')
    assert pvp_type.get_offset_size_format('AmpSF') is None

    assert pvp_type.get_offset_size_format('RcvEB') == (376, 16, 'd')
    pvp_type.RcvAntenna = None
    assert pvp_type.get_size() == 336
    assert pvp_type.get_offset_size_format('RcvEB') is None

    assert pvp_type.get_offset_size_format('TxACX') == (264, 24, 'd')
    pvp_type.TxAntenna = None
    assert pvp_type.get_offset_size_format('TxACX') is None

    assert pvp_type.get_offset_size_format('userpvp') == (392, 8, np.dtype('>i8').char)
    assert pvp_type.get_offset_size_format('NOTuserpvp') is None
    pvp_type.AddedPVP = None
    assert pvp_type.get_offset_size_format('userpvp') is None

