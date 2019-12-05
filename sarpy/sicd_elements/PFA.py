"""
The PFAType definition.
"""

import numpy
from numpy.linalg import norm

from .base import Serializable, DEFAULT_STRICT, _BooleanDescriptor, _FloatDescriptor, _SerializableDescriptor
from .blocks import Poly1DType, Poly2DType, XYZType

from sarpy.geometry import geocoords


__classification__ = "UNCLASSIFIED"


class STDeskewType(Serializable):
    """Parameters to describe image domain ST Deskew processing."""
    _fields = ('Applied', 'STDSPhasePoly')
    _required = _fields
    # descriptors
    Applied = _BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Parameter indicating if slow time (ST) Deskew Phase function has been applied.')  # type: bool
    STDSPhasePoly = _SerializableDescriptor(
        'STDSPhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Slow time deskew phase function to perform the ST / Kaz shift. Two-dimensional phase (cycles) '
                  'polynomial function of image range coordinate (variable 1) and '
                  'azimuth coordinate (variable 2).')  # type: Poly2DType


class PFAType(Serializable):
    """Parameters for the Polar Formation Algorithm."""
    _fields = (
        'FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2',
        'StDeskew')
    _required = ('FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2')
    # descriptors
    FPN = _SerializableDescriptor(
        'FPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Focus Plane unit normal (ECF). Unit vector FPN points away from the center of '
                  'the Earth.')  # type: XYZType
    IPN = _SerializableDescriptor(
        'IPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Image Formation Plane unit normal (ECF). Unit vector IPN points away from the '
                  'center of the Earth.')  # type: XYZType
    PolarAngRefTime = _FloatDescriptor(
        'PolarAngRefTime', _required, strict=DEFAULT_STRICT,
        docstring='Polar image formation reference time *in seconds*. Polar Angle = 0 at the reference time. '
                  'Measured relative to collection start. Note: Reference time is typically set equal to the SCP '
                  'COA time but may be different.')  # type: float
    PolarAngPoly = _SerializableDescriptor(
        'PolarAngPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Polar Angle (radians) as function of time '
                  'relative to Collection Start.')  # type: Poly1DType
    SpatialFreqSFPoly = _SerializableDescriptor(
        'SpatialFreqSFPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields the Spatial Frequency Scale Factor (KSF) as a function of Polar '
                  'Angle. Polar Angle(radians) -> KSF (dimensionless). Used to scale RF frequency (fx, Hz) to '
                  'aperture spatial frequency (Kap, cycles/m). `Kap = fx x (2/c) x KSF`. Kap is the effective spatial '
                  'frequency in the polar aperture.')  # type: Poly1DType
    Krg1 = _FloatDescriptor(
        'Krg1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum range spatial frequency (Krg) output from the polar to rectangular '
                  'resampling.')  # type: float
    Krg2 = _FloatDescriptor(
        'Krg2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum range spatial frequency (Krg) output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz1 = _FloatDescriptor(
        'Kaz1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum azimuth spatial frequency (Kaz) output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz2 = _FloatDescriptor(
        'Kaz2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum azimuth spatial frequency (Kaz) output from the polar to rectangular '
                  'resampling.')  # type: float
    StDeskew = _SerializableDescriptor(
        'StDeskew', STDeskewType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters to describe image domain ST Deskew processing.')  # type: STDeskewType

    def _derive_parameters(self, Grid, SCPCOA, GeoData):
        """
        Expected to be called from SICD parent.

        Parameters
        ----------
        Grid : sarpy.sicd_elements.GridType
        SCPCOA : sarpy.sicd_elements.SCPCOAType
        GeoData : sarpy.sicd_elements.GeoDataType

        Returns
        -------
        None
        """

        if self.PolarAngRefTime is None and SCPCOA.SCPTime is not None:
            self.PolarAngRefTime = SCPCOA.SCPTime

        if GeoData is not None and GeoData.SCP is not None and GeoData.SCP.ECF is not None and \
                SCPCOA.ARPPos is not None and SCPCOA.ARPVel is not None:
            SCP = GeoData.SCP.ECF.get_array()
            ETP = geocoords.wgs_84_norm(SCP)

            ARP = SCPCOA.ARPPos.get_array()
            LOS = (SCP - ARP)
            uLOS = LOS/norm(LOS)

            look = -1 if SCPCOA.SideOfTrack == 'R' else 1
            ARP_vel = SCPCOA.ARPVel.get_array()
            uSPZ = look*numpy.cross(ARP_vel, uLOS)
            uSPZ /= norm(uSPZ)
            if Grid is not None and Grid.ImagePlane is not None:
                if self.IPN is None:
                    if Grid.ImagePlane == 'SLANT':
                        self.IPN = XYZType(X=uSPZ[0], Y=uSPZ[1], Z=uSPZ[2])
                    elif Grid.ImagePlane == 'GROUND':
                        self.IPN = XYZType(X=ETP[0], Y=ETP[1], Z=ETP[2])
            elif self.IPN is None:
                self.IPN = XYZType(X=uSPZ[0], Y=uSPZ[1], Z=uSPZ[2])  # assuming slant -> most common

            if self.FPN is None:
                self.FPN = XYZType(X=ETP[0], Y=ETP[1], Z=ETP[2])

        # TODO: PolarAngPoly, SpatialFreqSFPoly - carried over from sicd.py line 1742
