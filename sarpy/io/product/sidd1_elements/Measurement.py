# -*- coding: utf-8 -*-
"""
The MeasurementType definition for SIDD 1.0.
"""

from typing import Union

from ..sidd2_elements.base import DEFAULT_STRICT

# noinspection PyProtectedMember
from ...complex.sicd_elements.base import Serializable, _SerializableDescriptor
from ..sidd2_elements.blocks import Poly2DType, RowColIntType, XYZPolyType
from ..sidd2_elements.Measurement import PolynomialProjectionType, GeographicProjectionType, \
    PlaneProjectionType, CylindricalProjectionType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class MeasurementType(Serializable):
    """
    Geometric SAR information required for measurement/geolocation.
    """

    _fields = (
        'PolynomialProjection', 'GeographicProjection', 'PlaneProjection', 'CylindricalProjection',
        'PixelFootprint', 'ARPFlag', 'ARPPoly')
    _required = ('PixelFootprint', 'ARPPoly')
    _collections_tags = {'ValidData': {'array': True, 'child_tag': 'Vertex'}}
    _numeric_format = {'ValidData': '0.16G'}
    # Descriptor
    PolynomialProjection = _SerializableDescriptor(
        'PolynomialProjection', PolynomialProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial pixel to ground. Should only used for sensor systems where the radar '
                  'geometry parameters are not recorded.')  # type: Union[None, PolynomialProjectionType]
    GeographicProjection = _SerializableDescriptor(
        'GeographicProjection', GeographicProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Geographic mapping of the pixel grid referred to as GGD in the '
                  'Design and Exploitation document.')  # type: Union[None, GeographicProjectionType]
    PlaneProjection = _SerializableDescriptor(
        'PlaneProjection', PlaneProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Planar representation of the pixel grid referred to as PGD in the '
                  'Design and Exploitation document.')  # type: Union[None, PlaneProjectionType]
    CylindricalProjection = _SerializableDescriptor(
        'CylindricalProjection', CylindricalProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Cylindrical mapping of the pixel grid referred to as CGD in the '
                  'Design and Exploitation document.')  # type: Union[None, CylindricalProjectionType]
    PixelFootprint = _SerializableDescriptor(
        'PixelFootprint', RowColIntType, _required, strict=DEFAULT_STRICT,
        docstring='Size of the image in pixels.')  # type: RowColType
    ARPPoly = _SerializableDescriptor(
        'ARPPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Center of aperture polynomial (units = m) based upon time into '
                  'the collect.')  # type: XYZPolyType

    def __init__(self, PolynomialProjection=None, GeographicProjection=None, PlaneProjection=None,
                 CylindricalProjection=None, ARPPoly=None, **kwargs):
        """

        Parameters
        ----------
        PolynomialProjection : PolynomialProjectionType
        GeographicProjection : GeographicProjectionType
        PlaneProjection : PlaneProjectionType
        CylindricalProjection : CylindricalProjectionType
        ARPPoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PolynomialProjection = PolynomialProjection
        self.GeographicProjection = GeographicProjection
        self.PlaneProjection = PlaneProjection
        self.CylindricalProjection = CylindricalProjection
        self.ARPPoly = ARPPoly
        super(MeasurementType, self).__init__(**kwargs)
