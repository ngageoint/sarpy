"""
The CompressionType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union

import xml.etree.ElementTree
import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import SerializableDescriptor, IntegerDescriptor, \
    FloatArrayDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT


# default_strict is set to False which is why invalid types don't throw an error for NumWaveletLevels, NumBands, or LayerInfo

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
        self.setLayerInfoType(LayerInfo)
        super(J2KSubtype, self).__init__(**kwargs)

    def setLayerInfoType(self, obj):
        """
        Since the LayerInfo is defined as an array of bit rates in the definition above and SarPy returns an
        element tree with the bit rates nested within different layers, this function is used to parse and return
        LayerInfo as an array of bit rates
        """
        # if LayerInfo is an ElementTree as expected, then it gets parsed to return an array of bit rates
        ET = xml.etree.ElementTree
        if isinstance( obj, ET.Element):
            numLayers = int(obj.attrib['numLayers'])
            if (numLayers == 0):
                return
            bitrates = numpy.zeros(numLayers)
          
            for i in range(numLayers):
                bitrates[i] = float(obj[i][0].text)
            self.LayerInfo = bitrates

        # if LayerInfo is a numpy.ndarray, a  list, or a tuple per the above definition, return LayerInfo
        elif isinstance(obj, (list, tuple, numpy.ndarray)):
            self.LayerInfo = obj 

        # none object handler since it states that LayerInfo can be None in the definition above and it isn't a required field
        elif obj is None:
            self.LayerInfo = None

        # throw an error if LayerInfo is an invalid type and an array of bitrates is unable to be generated
        else:
            raise TypeError(f'Invalid input type for LayerInfo: {type(obj)}. Must be an ElementTree, list, tuple, ndarray, or None.')

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