#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sarpy.io.phase_history.cphd1_elements import SupportArray


def test_support_array_supportarraycore(cphd):
    """Test SupporyArrayCore class"""
    sa = SupportArray.SupportArrayCore(
        Identifier=cphd.SupportArray.AntGainPhase[0].Identifier,
        ElementFormat=cphd.SupportArray.AntGainPhase[0].ElementFormat,
        X0=cphd.SupportArray.AntGainPhase[0].X0,
        Y0=cphd.SupportArray.AntGainPhase[0].Y0,
        XSS=cphd.SupportArray.AntGainPhase[0].XSS,
        YSS=cphd.SupportArray.AntGainPhase[0].YSS,
    )
    sa.NODATA = None
    assert sa.NODATA == None
    assert sa.get_nodata_as_int() is None
    assert sa.get_nodata_as_float() is None

    sa.NODATA = ET.fromstring("<NODATA>deadbeef12345678</NODATA>")
    assert sa.NODATA == "deadbeef12345678"

    sa.NODATA = "deadbeef12345678"
    assert sa.NODATA == "deadbeef12345678"

    sa.NODATA = b"deadbeef12345678"
    assert sa.NODATA == "deadbeef12345678"

    with pytest.raises(NotImplementedError):
        sa.NODATA = 1.0

    with pytest.raises(NotImplementedError):
        sa.NODATA = 1

    with pytest.raises(TypeError, match="Got unexpected type"):
        sa.NODATA = {1}

    with pytest.raises(NotImplementedError):
        sa.get_nodata_as_int()

    with pytest.raises(NotImplementedError):
        sa.get_nodata_as_float()

    fmt = sa.get_numpy_format()
    assert fmt[0] == np.dtype(">f4")
    assert fmt[1] == 2


def test_support_array_supportarraytype(cphd):
    """Test SupporyArrayType class"""
    sa = SupportArray.SupportArrayType(
        IAZArray=cphd.SupportArray.IAZArray,
        AntGainPhase=cphd.SupportArray.AntGainPhase,
        DwellTimeArray=cphd.SupportArray.DwellTimeArray,
        AddedSupportArray=cphd.SupportArray.AddedSupportArray,
    )

    iaz_array = sa.find_support_array("iaz_id0")
    assert iaz_array.Identifier == cphd.SupportArray.IAZArray[0].Identifier

    ant_gain_phase = sa.find_support_array("transmit_array")
    assert ant_gain_phase.Identifier == cphd.SupportArray.AntGainPhase[0].Identifier

    dwell_time_array = sa.find_support_array("dta_id0")
    assert dwell_time_array.Identifier == cphd.SupportArray.DwellTimeArray[0].Identifier

    added_support_array = sa.find_support_array("added_support_array0")
    assert (
        added_support_array.Identifier
        == cphd.SupportArray.AddedSupportArray[0].Identifier
    )

    with pytest.raises(
        KeyError, match="Identifier IAZArrayType not associated with a support array"
    ):
        iaz_array = sa.find_support_array("IAZArrayType")

