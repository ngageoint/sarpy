"""
The DataType definition for CPHD 0.3.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import List

from sarpy.io.phase_history.cphd1_elements.base import DEFAULT_STRICT
from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import StringEnumDescriptor, IntegerDescriptor, SerializableListDescriptor


class ArraySizeType(Serializable):
    """
    Parameters that define the array sizes.
    """

    _fields = ('NumVectors', 'NumSamples')
    _required = ('NumVectors', 'NumSamples')
    # descriptors
    NumVectors = IntegerDescriptor(
        'NumVectors', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of slow time vectors in the PHD array in this channel.')  # type: int
    NumSamples = IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of samples per vector in the PHD array in this channel.')  # type: int

    def __init__(self, NumVectors=None, NumSamples=None, **kwargs):
        """

        Parameters
        ----------
        NumVectors : int
        NumSamples : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumVectors = NumVectors
        self.NumSamples = NumSamples
        super(ArraySizeType, self).__init__(**kwargs)


class DataType(Serializable):
    """
    Parameters that describe binary data components contained in the product.
    """

    _fields = ('SampleType', 'NumCPHDChannels', 'NumBytesVBP', 'ArraySize')
    _required = ('SampleType', 'NumBytesVBP', 'ArraySize')
    _collections_tags = {'ArraySize': {'array': False, 'child_tag': 'ArraySize'}}
    # descriptors
    SampleType = StringEnumDescriptor(
        'SampleType', ("RE32F_IM32F", "RE16I_IM16I", "RE08I_IM08I"), _required, strict=True,
        docstring="Indicates the PHD sample format of the PHD array(s). All arrays "
                  "have the sample type. Real and imaginary components stored in adjacent "
                  "bytes, real component stored first.")  # type: str
    NumBytesVBP = IntegerDescriptor(
        'NumBytesVBP', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of bytes per set of Vector Based Parameters.')  # type: int
    ArraySize = SerializableListDescriptor(
        'ArraySize', ArraySizeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='CPHD array size parameters.')  # type: List[ArraySizeType]

    def __init__(self, SampleType=None, NumBytesVBP=None, ArraySize=None, **kwargs):
        """

        Parameters
        ----------
        SampleType : str
        NumBytesVBP : int
        ArraySize : List[ArraySizeType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SampleType = SampleType
        self.NumBytesVBP = NumBytesVBP
        self.ArraySize = ArraySize
        super(DataType, self).__init__(**kwargs)

    @property
    def NumCPHDChannels(self):
        """
        int: The number of CPHD channels.
        """

        if self.ArraySize is None:
            return 0
        return len(self.ArraySize)
