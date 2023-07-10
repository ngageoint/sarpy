#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import Antenna
from sarpy.io.complex.sicd_elements import blocks


@pytest.fixture
def tx_ant_param(sicd, kwargs):
    return Antenna.AntParamType(XAxisPoly=sicd.Antenna.Tx.XAxisPoly,
                                YAxisPoly=sicd.Antenna.Tx.YAxisPoly,
                                FreqZero=sicd.Antenna.Tx.FreqZero,
                                EB=sicd.Antenna.Tx.EB,
                                Array=sicd.Antenna.Tx.Array,
                                Elem=sicd.Antenna.Tx.Elem,
                                GainBSPoly=sicd.Antenna.Tx.GainBSPoly,
                                EBFreqShift=True,
                                MLFreqDilation=sicd.Antenna.Tx.MLFreqDilation,
                                **kwargs)


@pytest.fixture
def rcv_ant_param(sicd, kwargs):
    return Antenna.AntParamType(XAxisPoly=sicd.Antenna.Rcv.XAxisPoly,
                                YAxisPoly=sicd.Antenna.Rcv.YAxisPoly,
                                FreqZero=sicd.Antenna.Rcv.FreqZero,
                                EB=sicd.Antenna.Rcv.EB,
                                Array=sicd.Antenna.Rcv.Array,
                                Elem=sicd.Antenna.Rcv.Elem,
                                GainBSPoly=sicd.Antenna.Rcv.GainBSPoly,
                                EBFreqShift=True,
                                MLFreqDilation=sicd.Antenna.Rcv.MLFreqDilation,
                                **kwargs)


@pytest.fixture
def twoway_ant_param(sicd, kwargs):
    return Antenna.AntParamType(XAxisPoly=sicd.Antenna.TwoWay.XAxisPoly,
                                YAxisPoly=sicd.Antenna.TwoWay.YAxisPoly,
                                FreqZero=sicd.Antenna.TwoWay.FreqZero,
                                EB=sicd.Antenna.TwoWay.EB,
                                Array=sicd.Antenna.TwoWay.Array,
                                Elem=sicd.Antenna.TwoWay.Elem,
                                GainBSPoly=sicd.Antenna.TwoWay.GainBSPoly,
                                EBFreqShift=True,
                                MLFreqDilation=sicd.Antenna.TwoWay.MLFreqDilation,
                                **kwargs)


def test_antenna_ebtype(kwargs):
    x_poly = blocks.Poly1DType(Coefs=[10.5, 5.1, 1.2, 0.2])
    y_poly = blocks.Poly1DType(Coefs=[5.1, 1.2, 0.2])

    antenna_eb = Antenna.EBType(DCXPoly=x_poly, DCYPoly=y_poly)
    assert antenna_eb.DCXPoly == x_poly
    assert antenna_eb.DCYPoly == y_poly
    assert not hasattr(antenna_eb, "_xml_ns")
    assert not hasattr(antenna_eb, "_xml_ns_key")

    # Init with kwargs
    antenna_eb = Antenna.EBType(DCXPoly=x_poly, DCYPoly=y_poly, **kwargs)
    assert antenna_eb._xml_ns == kwargs["_xml_ns"]
    assert antenna_eb._xml_ns_key == kwargs["_xml_ns_key"]

    assert np.all(antenna_eb(0) == [x_poly.Coefs[0], y_poly.Coefs[0]])

    antenna_eb = Antenna.EBType(DCXPoly=None, DCYPoly=y_poly)
    assert antenna_eb(0) is None


def test_antenna_antparamtype(tx_ant_param, sicd, kwargs):
    assert tx_ant_param._xml_ns == kwargs["_xml_ns"]
    assert tx_ant_param._xml_ns_key == kwargs["_xml_ns_key"]

    shift = 100000
    tx_ant_param._apply_reference_frequency(shift)
    assert tx_ant_param.FreqZero == sicd.Antenna.Tx.FreqZero + shift


def test_antenna_anttype(tx_ant_param, rcv_ant_param, twoway_ant_param, sicd, kwargs):
    antenna = Antenna.AntennaType(Tx=tx_ant_param, Rcv=rcv_ant_param, TwoWay=twoway_ant_param, **kwargs)
    assert antenna._xml_ns == kwargs["_xml_ns"]
    assert antenna._xml_ns_key == kwargs["_xml_ns_key"]

    shift = 100000
    antenna._apply_reference_frequency(shift)
    assert antenna.Tx.FreqZero == sicd.Antenna.Tx.FreqZero + shift
    assert antenna.Rcv.FreqZero == sicd.Antenna.Rcv.FreqZero + shift
    assert antenna.TwoWay.FreqZero == sicd.Antenna.TwoWay.FreqZero + shift
