"""
The PositionType definition.
"""

import numpy

from typing import List, Union

from ._base import Serializable, DEFAULT_STRICT, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from ._blocks import XYZPolyType, XYZPolyAttributeType


__classification__ = "UNCLASSIFIED"


class PositionType(Serializable):
    """The details for platform and ground reference positions as a function of time since collection start."""
    _fields = ('ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC')
    _required = ('ARPPoly',)
    _collections_tags = {'RcvAPC': {'array': True, 'child_tag': 'RcvAPCPoly'}}

    # descriptors
    ARPPoly = _SerializableDescriptor(
        'ARPPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Aperture Reference Point (ARP) position polynomial in ECF as a function of elapsed '
                  'seconds since start of collection.')  # type: XYZPolyType
    GRPPoly = _SerializableDescriptor(
        'GRPPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Ground Reference Point (GRP) position polynomial in ECF as a function of elapsed '
                  'seconds since start of collection.')  # type: XYZPolyType
    TxAPCPoly = _SerializableDescriptor(
        'TxAPCPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Transmit Aperture Phase Center (APC) position polynomial in ECF as a function of '
                  'elapsed seconds since start of collection.')  # type: XYZPolyType
    RcvAPC = _SerializableArrayDescriptor(
        'RcvAPC', XYZPolyAttributeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Receive Aperture Phase Center polynomials array. '
                  'Each polynomial has output in ECF, and represents a function of elapsed seconds since start of '
                  'collection.')  # type: Union[numpy.ndarray, List[XYZPolyAttributeType]]
