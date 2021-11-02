"""
The CompressionType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union

import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import SerializableDescriptor, IntegerDescriptor, \
    FloatArrayDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT


class J2KSubtype(Serializable):
    """
    The Jpeg 2000 subtype.
    """
    _fields = ('NumWaveletLevels', 'NumBands', 'LayerInfo')
    _required = ('NumWaveletLevels', 'NumBands')
    _collections_tags = {'LayerInfo': {'array': True, 'child_tag': 'Bitrate', 'size_attribute': 'numLayers'}}
    _numeric_format = {'LayerInfo': FLOAT_FORMAT}
    # Descriptor
    NumWaveletLevels = IntegerDescriptor(
        'NumWaveletLevels', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: int
    NumBands = IntegerDescriptor(
        'NumBands', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: int
    LayerInfo = FloatArrayDescriptor(
        'LayerInfo', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Original Layer Information. This is an array of bit rate target associated with each '
                  'layer. It may happen that the bit rate was not achieved due to data characteristics. '
                  '**Note -** for JPEG 2000 numerically loss-less quality, the bit rate for the final layer is '
                  'an expected value, based on performance.')  # type: Union[None, numpy.ndarray]

    def __init__(self, NumWaveletLevels=None, NumBands=None, LayerInfo=None, **kwargs):
        """

        Parameters
        ----------
        NumWaveletLevels : int
        NumBands : int
        LayerInfo : None|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumWaveletLevels = NumWaveletLevels
        self.NumBands = NumBands
        self.LayerInfo = LayerInfo
        super(J2KSubtype, self).__init__(**kwargs)


class J2KType(Serializable):
    """
    Jpeg 2000 parameters.
    """

    _fields = ('Original', 'Parsed')
    _required = ('Original', )
    # Descriptor
    Original = SerializableDescriptor(
        'Original', J2KSubtype, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: J2KSubtype
    Parsed = SerializableDescriptor(
        'Parsed', J2KSubtype, _required, strict=DEFAULT_STRICT,
        docstring='Conditional fields that exist only for parsed images.')  # type: Union[None, J2KSubtype]

    def __init__(self, Original=None, Parsed=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Original = Original
        self.Parsed = Parsed
        super(J2KType, self).__init__(**kwargs)


class CompressionType(Serializable):
    """
    Contains information regarding any compression that has occurred to the image data.
    """

    _fields = ('J2K', )
    _required = ('J2K', )
    # Descriptor
    J2K = SerializableDescriptor(
        'J2K', J2KType, _required, strict=DEFAULT_STRICT,
        docstring='Block describing details of JPEG 2000 compression.')  # type: J2KType

    def __init__(self, J2K=None, **kwargs):
        """

        Parameters
        ----------
        J2K : J2KType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.J2K = J2K
        super(CompressionType, self).__init__(**kwargs)
