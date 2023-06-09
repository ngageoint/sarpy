#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import copy
import re

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import SCPCOA


@pytest.fixture()
def scpcoa(sicd, kwargs):
    return SCPCOA.SCPCOAType(
        SCPTime=sicd.SCPCOA.SCPTime,
        ARPPos=sicd.SCPCOA.ARPPos,
        ARPVel=sicd.SCPCOA.ARPVel,
        ARPAcc=sicd.SCPCOA.ARPAcc,
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
        **kwargs,
    )


def test_scpcoa_geometrycalculator(sicd, caplog):
    # Smoke test
    geom_calc = SCPCOA.GeometryCalculator(
        SCP=sicd.GeoData.SCP.ECF.get_array(),
        ARPPos=sicd.SCPCOA.ARPPos.get_array(),
        ARPVel=sicd.SCPCOA.ARPVel.get_array(),
    )

    bad_vector = np.asarray([1e-7, 0, 0])
    assert np.allclose(geom_calc._make_unit(bad_vector), [1, 0, 0])
    assert f"The input vector to be normalized has norm" in caplog.text
    rov = geom_calc.ROV
    assert rov is not None
    sot = geom_calc.SideOfTrack
    assert sot == "L"
    slant_range = geom_calc.SlantRange
    assert slant_range > 0.0
    ground_range = geom_calc.GroundRange
    assert ground_range > 0.0
    dca = geom_calc.DopplerConeAng
    assert 0.0 <= dca <= 180.0
    graze = geom_calc.GrazeAng
    assert 0.0 <= graze <= 90.0
    incidence = geom_calc.IncidenceAng
    assert 0.0 <= incidence <= 90.0
    graze, incidence = geom_calc.get_graze_and_incidence()
    assert 0.0 <= graze <= 90.0
    assert 0.0 <= incidence <= 90.0
    twist = geom_calc.TwistAng
    assert -90.0 <= twist <= 90.0
    squint = geom_calc.SquintAngle
    assert -90.0 <= squint <= 90.0
    slope = geom_calc.SlopeAng
    assert 0.0 < slope <= 90.0
    azim = geom_calc.AzimAng
    assert 0.0 <= azim <= 360.0
    layover = geom_calc.LayoverAng
    assert 0.0 <= layover <= 360.0
    layover = geom_calc.get_layover()
    assert np.all(layover is not None)
    shadow = geom_calc.get_shadow()
    assert np.all(shadow is not None)


def test_scpcoa(scpcoa, kwargs):
    # Smoke test
    assert scpcoa._xml_ns == kwargs["_xml_ns"]
    assert scpcoa._xml_ns_key == kwargs["_xml_ns_key"]


def test_scpcoa_look(scpcoa):
    assert scpcoa.look is not None
    scpcoa.SideOfTrack = None
    assert scpcoa.look is None


def test_scpcoa_rov(sicd, scpcoa):
    assert scpcoa.ROV is None
    scpcoa._derive_geometry_parameters(GeoData=sicd.GeoData, overwrite=True)
    assert scpcoa.ROV is not None


def test_scpcoa_thetadot(sicd, scpcoa):
    assert scpcoa.ThetaDot is None
    scpcoa._derive_geometry_parameters(GeoData=sicd.GeoData, overwrite=True)
    assert scpcoa.ThetaDot is not None


def test_scpcoa_multipathground(scpcoa):
    assert scpcoa.MultipathGround is not None
    scpcoa.GrazeAng = None
    assert scpcoa.MultipathGround is None


def test_scpcoa_multipath(scpcoa):
    assert scpcoa.Multipath is not None
    scpcoa.AzimAng = None
    assert scpcoa.Multipath is None


def test_scpcoa_shadow(sicd, scpcoa):
    assert scpcoa.Shadow is None
    scpcoa._derive_geometry_parameters(GeoData=sicd.GeoData, overwrite=True)
    assert scpcoa.Shadow is not None


def test_scpcoa_shadowmagnitude(sicd, scpcoa):
    assert scpcoa.ShadowMagnitude is None
    scpcoa._derive_geometry_parameters(GeoData=sicd.GeoData, overwrite=True)
    assert scpcoa.ShadowMagnitude is not None


def test_scpcoa_squint(sicd, scpcoa):
    assert scpcoa.Squint is None
    scpcoa._derive_geometry_parameters(GeoData=sicd.GeoData, overwrite=True)
    assert scpcoa.Squint is not None


def test_scpcoa_layovermagnitude(sicd, scpcoa):
    assert scpcoa.LayoverMagnitude is None
    scpcoa._derive_geometry_parameters(GeoData=sicd.GeoData, overwrite=True)
    assert scpcoa.LayoverMagnitude is not None


def test_scpcoa_derivescptime(sicd, scpcoa, tol):
    scp_time = scpcoa.SCPTime
    scpcoa.SCPTime = 0.0

    # Do nothing path
    scpcoa._derive_scp_time(Grid=None)
    assert scpcoa.SCPTime == 0.0

    # Another do nothing path
    scpcoa._derive_scp_time(Grid=sicd.Grid)
    assert scpcoa.SCPTime == 0.0

    scpcoa._derive_scp_time(Grid=sicd.Grid, overwrite=True)
    assert scpcoa.SCPTime == pytest.approx(scp_time, abs=tol)


def test_scpcoa_deriveposition(sicd, scpcoa, tol):
    arp_pos = scpcoa.ARPPos.get_array()
    arp_vel = scpcoa.ARPVel.get_array()
    arp_acc = scpcoa.ARPAcc.get_array()
    scpcoa.ARPPos = None
    scpcoa.ARPVel = None
    scpcoa.ARPAcc = None

    # Do nothing path
    scpcoa._derive_position(Position=None)
    assert scpcoa.ARPPos == None
    assert scpcoa.ARPVel == None
    assert scpcoa.ARPAcc == None

    scpcoa._derive_position(Position=sicd.Position, overwrite=True)
    assert np.all(scpcoa.ARPPos.get_array() == pytest.approx(arp_pos, abs=tol))
    assert np.all(scpcoa.ARPVel.get_array() == pytest.approx(arp_vel, abs=tol))
    assert np.all(scpcoa.ARPAcc.get_array() == pytest.approx(arp_acc, abs=tol))


def test_scpcoa_derivegeometry(sicd, scpcoa, tol):
    scpcoa_copy = copy.copy(scpcoa)

    scpcoa_copy._ROV = None
    scpcoa_copy.SideOfTrack = None
    scpcoa_copy.SlantRange = None
    scpcoa_copy.GroundRange = None
    scpcoa_copy.DopplerConeAng = None
    scpcoa_copy.GrazeAng = None
    scpcoa_copy.IncidenceAng = None
    scpcoa_copy.TwistAng = None
    scpcoa_copy._squint = None
    scpcoa_copy.SlopeAng = None
    scpcoa_copy.AzimAng = None
    scpcoa_copy.LayoverAng = None

    # Do nothing path
    scpcoa_copy._derive_geometry_parameters(GeoData=None)
    assert scpcoa_copy._ROV == None
    assert scpcoa_copy.SideOfTrack == None
    assert scpcoa_copy.SlantRange == None
    assert scpcoa_copy.GroundRange == None
    assert scpcoa_copy.DopplerConeAng == None
    assert scpcoa_copy.GrazeAng == None
    assert scpcoa_copy.IncidenceAng == None
    assert scpcoa_copy.TwistAng == None
    assert scpcoa_copy._squint == None
    assert scpcoa_copy.SlopeAng == None
    assert scpcoa_copy.AzimAng == None
    assert scpcoa_copy.LayoverAng == None

    scpcoa_copy._derive_geometry_parameters(GeoData=sicd.GeoData, overwrite=True)
    assert scpcoa_copy._ROV is not None
    assert scpcoa_copy.SideOfTrack == pytest.approx(scpcoa.SideOfTrack, abs=tol)
    assert scpcoa_copy.SlantRange == pytest.approx(scpcoa.SlantRange, abs=tol)
    assert scpcoa_copy.GroundRange == pytest.approx(scpcoa.GroundRange, abs=tol)
    assert scpcoa_copy.DopplerConeAng == pytest.approx(scpcoa.DopplerConeAng, abs=tol)
    assert scpcoa_copy.GrazeAng == pytest.approx(scpcoa.GrazeAng, abs=tol)
    assert scpcoa_copy.IncidenceAng == pytest.approx(scpcoa.IncidenceAng, abs=tol)
    assert scpcoa_copy.TwistAng == pytest.approx(scpcoa.TwistAng, abs=tol)
    assert scpcoa_copy._squint is not None
    assert scpcoa_copy.SlopeAng == pytest.approx(scpcoa.SlopeAng, abs=tol)
    assert scpcoa_copy.AzimAng == pytest.approx(scpcoa.AzimAng, abs=tol)
    assert scpcoa_copy.LayoverAng == pytest.approx(scpcoa.LayoverAng, abs=tol)


def test_scpcoa_rederive(sicd, scpcoa):
    # Smoke test
    scpcoa.rederive(Grid=sicd.Grid, Position=sicd.Position, GeoData=sicd.GeoData)


def test_scpcoa_checkvalues(sicd, scpcoa):
    # Do nothing path
    assert scpcoa.check_values(GeoData=None)

    # Smoke test
    assert scpcoa.check_values(GeoData=sicd.GeoData)


def test_scpcoa_checkvalues_error1(sicd, scpcoa, caplog):
    scpcoa.SideOfTrack = "R"
    assert not scpcoa.check_values(GeoData=sicd.GeoData)
    assert "SideOfTrack is expected to be L, and is populated as R" in caplog.text


def test_scpcoa_checkvalues_error2(sicd, scpcoa, caplog):
    scpcoa.SlantRange = 1000000.0
    scpcoa.GroundRange = 1000000.0
    assert not scpcoa.check_values(GeoData=sicd.GeoData)
    assert (
        f"attribute SlantRange is expected to have value {np.round(sicd.SCPCOA.SlantRange, 10)}, but is populated as 1000000.0"
        in caplog.text
    )
    assert (
        f"attribute GroundRange is expected to have value {np.round(sicd.SCPCOA.GroundRange, 10)}, but is populated as 1000000.0"
        in caplog.text
    )


def test_scpcoa_checkvalues_error3(sicd, scpcoa, caplog):
    scpcoa.DopplerConeAng = 360.0
    scpcoa.GrazeAng = 360.0
    scpcoa.IncidenceAng = 360.0
    scpcoa.TwistAng = 360.0
    scpcoa.SlopeAng = 360.0
    scpcoa.AzimAng = 360.0
    scpcoa.LayoverAng = 360.0
    assert not scpcoa.check_values(GeoData=sicd.GeoData)

    assert re.search(
        r"attribute DopplerConeAng is expected to have value (\d+\.\d+), but is populated as 360.0",
        caplog.text,
    )
    assert re.search(
        r"attribute GrazeAng is expected to have value (\d+\.\d+), but is populated as 360.0",
        caplog.text,
    )
    assert re.search(
        r"attribute IncidenceAng is expected to have value (\d+\.\d+), but is populated as 360.0",
        caplog.text,
    )
    assert re.search(
        r"attribute TwistAng is expected to have value (\d+\.\d+), but is populated as 360.0",
        caplog.text,
    )
    assert re.search(
        r"attribute SlopeAng is expected to have value (\d+\.\d+), but is populated as 360.0",
        caplog.text,
    )
    assert re.search(
        r"attribute AzimAng is expected to have value (\d+\.\d+), but is populated as 360.0",
        caplog.text,
    )
    assert re.search(
        r"attribute LayoverAng is expected to have value (\d+\.\d+), but is populated as 360.0",
        caplog.text,
    )
