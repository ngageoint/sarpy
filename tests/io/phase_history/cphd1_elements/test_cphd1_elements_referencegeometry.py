#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.phase_history.cphd1_elements import ReferenceGeometry
from sarpy.io.complex.sicd_elements.blocks import XYZType


def test_reference_geometry_monostatic():
    """Test MonostaticType class"""
    monostatic_type = ReferenceGeometry.MonostaticType(
        ARPPos=XYZType(7.2e6, 2.6e5, 1.5e6),
        ARPVel=XYZType(340, -7330, -400),
        TwistAngle=10.0,
        SlopeAngle=30.0,
        LayoverAngle=350.0,
        SideOfTrack='R',
        SlantRange=1.7e6,
        GroundRange=1.3e6,
        DopplerConeAngle=70.0,
        GrazeAngle=30.0,
        IncidenceAngle=60.0,
        AzimuthAngle=10.0,
    )
    assert monostatic_type.look == -1
    monostatic_type.SideOfTrack = None
    assert monostatic_type.look is None

    assert 0.0 <= monostatic_type.Multipath <= 360.0
    monostatic_type.TwistAngle = None
    assert monostatic_type.Multipath is None

    assert 0.0 <= monostatic_type.Shadow <= 360.0

