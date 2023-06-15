#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import copy

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import Position
from sarpy.io.complex.sicd_elements import SCPCOA


@pytest.fixture()
def position(sicd, kwargs):
    return Position.PositionType(
        ARPPoly=sicd.Position.ARPPoly,
        GRPPoly=sicd.Position.GRPPoly,
        TxAPCPoly=sicd.Position.TxAPCPoly,
        RcvAPC=sicd.Position.RcvAPC,
        **kwargs,
    )


def test_position_positiontype(sicd, position, kwargs):
    # Smoke test
    assert position._xml_ns == kwargs["_xml_ns"]
    assert position._xml_ns_key == kwargs["_xml_ns_key"]
    assert position.ARPPoly == sicd.Position.ARPPoly
    assert position.GRPPoly == sicd.Position.GRPPoly
    assert position.TxAPCPoly == sicd.Position.TxAPCPoly
    assert position.RcvAPC == sicd.Position.RcvAPC


def test_position_derivearppoly(sicd, position):
    scpcoa = SCPCOA.SCPCOAType(
        SCPTime=sicd.SCPCOA.SCPTime,
        ARPPos=sicd.SCPCOA.ARPPos,
        ARPVel=sicd.SCPCOA.ARPVel,
        ARPAcc=None,
        SideOfTrack=sicd.SCPCOA.SideOfTrack,
        SlantRange=sicd.SCPCOA.SlantRange,
        GroundRange=sicd.SCPCOA.GroundRange,
        DopplerConeAng=sicd.SCPCOA.DopplerConeAng,
        GrazeAng=sicd.SCPCOA.GrazeAng,
        IncidenceAng=sicd.SCPCOA.IncidenceAng,
        TwistAng=sicd.SCPCOA.TwistAng,
        SlopeAng=sicd.SCPCOA.SlopeAng,
        AzimAng=sicd.SCPCOA.AzimAng,
        LayoverAng=sicd.SCPCOA.LayoverAng,
    )
    arp_poly = copy.copy(position.ARPPoly)

    # Do nothing path
    position._derive_arp_poly(SCPCOA=scpcoa)
    assert np.all(position.ARPPoly.X.Coefs == arp_poly.X.Coefs)
    assert np.all(position.ARPPoly.Y.Coefs == arp_poly.Y.Coefs)
    assert np.all(position.ARPPoly.Z.Coefs == arp_poly.Z.Coefs)

    # Another do nothing path
    position.ARPPoly = None
    position._derive_arp_poly(SCPCOA=None)
    assert position.ARPPoly is None

    position._derive_arp_poly(SCPCOA=scpcoa)
    assert position.ARPPoly is not None
    position.ARPPoly.X.order1 == 2
    position.ARPPoly.Y.order1 == 2
    position.ARPPoly.Z.order1 == 2


def test_position_validitycheck(position, caplog):
    assert position._basic_validity_check()

    # Force the error condition
    position.ARPPoly.X.Coefs = position.ARPPoly.X.Coefs[0:1]
    assert not position._basic_validity_check()
    assert "ARPPoly should be order at least 1 in each component" in caplog.text
